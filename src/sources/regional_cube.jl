"""
    regional_cube.jl

Unified Environmental and Biogeochemical Data Cube assimilation engine for Tethys.

Provides declarative configuration, ingestion, harmonized spatial-vertical-temporal
regridding, and serialization for multi-parameter marine environmental cubes.
Supports:
- Climatological cubes (`:climatology`) for equilibrium baseline modeling
- Full transient historical time series (`:timeseries`) for dynamic hindcasts
- Hybrid dual-resolution archives (`:hybrid`)
"""

using Dates
using TOML
using Statistics
using ..Types: GeoDataset, GeoArray, Dimension, CoordinateSystem
using ..StorageRegistry
using ..Coordinates: bounding_box
import ..open_geostorage, ..close_geostorage, ..write_storage_variable!

export RegionalCubeConfig,
       load_cube_config,
       standard_ocean_depths,
       assimilate_regional_cube,
       build_cube_hsi_evaluator,
       evaluate_bioenergetic_scope

"""
    RegionalCubeConfig

Specification for assimilating a unified regional environmental data cube.

# Fields
- `region_name::String`: Unique human-readable regional identifier (e.g., "scotian_shelf").
- `lon_range::Tuple{Float64, Float64}`: Western and eastern longitudinal bounds (degrees east).
- `lat_range::Tuple{Float64, Float64}`: Southern and northern latitudinal bounds (degrees north).
- `resolution_deg::Float64`: Target horizontal grid resolution in degrees (default ~0.08333°).
- `depth_levels::Vector{Float64}`: Oceanographic depth levels in meters.
- `time_mode::Symbol`: `:climatology`, `:timeseries`, or `:hybrid`.
- `time_range::Union{Nothing, Tuple{Date, Date}}`: Start and end date for `:timeseries` mode.
- `time_step::Symbol`: `:climatology_monthly`, `:monthly`, or `:daily`.
- `climatology_months::Vector{Int}`: Climatological months to ingest (`1:12`).
- `sources::Dict{Symbol, Any}`: Data source catalog mapping (:bathymetry, :physics, :bgc).
- `variables::Vector{Symbol}`: Target scientific variables to assimilate.
- `output_path::String`: Target filesystem destination (e.g. "scotian_shelf_cube.zarr").
- `chunk_sizes::NamedTuple`: Chunking dimensions for tensor backend.
- `compressor::Symbol`: Compression codec (e.g. `:zstd`).
"""
Base.@kwdef struct RegionalCubeConfig
    region_name::String
    lon_range::Tuple{Float64, Float64}
    lat_range::Tuple{Float64, Float64}
    # `resolution_km` is preferred: it is the unit a shelf is actually described in, and a
    # degree of longitude is not a degree of latitude. `resolution_deg` is used when
    # `resolution_km` is not given.
    resolution_km::Union{Nothing, Real} = nothing
    resolution_deg::Float64 = 1.0 / 12.0
    depth_levels::Vector{Float64} = standard_ocean_depths()
    time_mode::Symbol = :climatology
    time_range::Union{Nothing, Tuple{Date, Date}} = nothing
    time_step::Symbol = :climatology_monthly
    climatology_months::Vector{Int} = collect(1:12)
    variables::Vector{Symbol} = [
        :elevation, :bottom_depth, :temperature, :salinity,
        :u, :v, :dissolved_oxygen, :oxygen_saturation,
        :ph, :nitrate, :phosphate, :silicate, :chlorophyll_a
    ]
    output_path::String = joinpath(store_root(), "cubes", "$(region_name)_environmental")
    chunk_sizes::NamedTuple = (lon = 60, lat = 60, depth = 25, time = 12)
    compressor::Symbol = :zstd
    # Per-variable source choice. This is the level at which a decision is actually made:
    # a run can take temperature from a reanalysis and oxygen from a climatology, and a
    # source that does not carry a variable simply does not supply it rather than being
    # substituted silently.
    #
    # Keyed by variable name. The value is the source key from DATA_SOURCES; the tier it
    # sits at is a property of that source and lives in the manifest, so it is validated
    # at load time rather than duplicated here.
    sources::Dict{Symbol, Symbol} = Dict{Symbol, Symbol}()

end

"""
    source_of(config::RegionalCubeConfig, variable::Symbol) -> Symbol

Which source a cube variable is taken from.

The config's per-variable `sources` entry wins, so a run can say "take temperature from
GLORYS and oxygen from WOA23". Failing that, the default in `CUBE_VARIABLE_SOURCES` applies.
Both are validated against `DATA_SOURCES`, so a name that is not a declared source is an
error naming what is available rather than a silent omission.
"""
function source_of(config::RegionalCubeConfig, variable::Symbol)
    key = get(config.sources, variable, nothing)
    if key === nothing
        key = get(CUBE_VARIABLE_SOURCES, variable, nothing)
    end
    key === nothing && error(
        "No source is declared for cube variable $(variable). Declared defaults: " *
        "$(sort(collect(keys(CUBE_VARIABLE_SOURCES)))). Either map it there or set " *
        "[sources] $(variable) = \"...\" in the config.")
    haskey(Manifest.DATA_SOURCES, key) ||
        error("Variable $(variable) resolves to source $(key), which is not declared in " *
              "DATA_SOURCES. Declared: $(sort(collect(keys(Manifest.DATA_SOURCES)))).")
    return key
end

"""
    standard_ocean_depths() -> Vector{Float64}

Return 47 standard oceanographic depth levels (m), surface-refined down to 4000 m.
"""
function standard_ocean_depths()::Vector{Float64}
    return Float64[
        0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
        60.0, 70.0, 80.0, 90.0, 100.0, 125.0, 150.0, 175.0, 200.0,
        225.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 600.0, 700.0,
        800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0,
        1750.0, 2000.0, 2250.0, 2500.0, 2750.0, 3000.0, 3250.0, 3500.0,
        3750.0, 4000.0
    ]
end

"""
    load_cube_config(path::AbstractString) -> RegionalCubeConfig

Parse a TOML configuration file into a validated `RegionalCubeConfig` instance.
"""
function load_cube_config(path::AbstractString)::RegionalCubeConfig
    isfile(path) || error("Configuration file not found: $(path)")
    raw = TOML.parsefile(path)
    
    rname = get(raw, "region_name", "custom_region")
    bounds = get(raw, "domain", raw)
    lon_r = (Float64(bounds["lon_min"]), Float64(bounds["lon_max"]))
    lat_r = (Float64(bounds["lat_min"]), Float64(bounds["lat_max"]))
    # `resolution_km` wins over `resolution_deg` when both are present: kilometres is the
    # unit a shelf is described in, and a degree grid is not square.
    res_km = haskey(raw, "resolution_km") ? Float64(raw["resolution_km"]) : nothing
    res_deg = Float64(get(raw, "resolution_deg", 1.0 / 12.0))
    
    depths = if haskey(raw, "depth_levels")
        Float64.(raw["depth_levels"])
    else
        standard_ocean_depths()
    end

    t_mode = Symbol(get(raw, "time_mode", "climatology"))
    t_step = Symbol(get(raw, "time_step", "climatology_monthly"))
    
    t_range = if haskey(raw, "start_date") && haskey(raw, "end_date")
        (Date(raw["start_date"]), Date(raw["end_date"]))
    else
        nothing
    end
    
    c_months = if haskey(raw, "climatology_months")
        Int.(raw["climatology_months"])
    else
        collect(1:12)
    end
    
    # A cube's variables and their sources are declared in code by default: the config says
    # which variables a run wants, `CUBE_VARIABLE_SOURCES` says where each comes from, and a
    # `[sources]` block in the config overrides that per variable.
    vars = if haskey(raw, "variables")
        Symbol.(raw["variables"])
    else
        [
            :elevation, :bottom_depth, :temperature, :salinity,
            :u, :v, :dissolved_oxygen, :oxygen_saturation,
            :ph, :nitrate, :phosphate, :silicate, :chlorophyll_a
        ]
    end

    unknown = [v for v in vars if !haskey(CUBE_VARIABLE_SOURCES, v) &&
               !haskey(get(raw, "sources", Dict{String,Any}()), string(v))]
    isempty(unknown) || error(
        "Config declares variable(s) $(unknown), which has no source in either " *
        "CUBE_VARIABLE_SOURCES or the config's [sources] block. Declared defaults: " *
        "$(sort(collect(keys(CUBE_VARIABLE_SOURCES)))).")

    srcs = Dict{Symbol, Any}()
    if haskey(raw, "sources")
        for (k, v) in raw["sources"]
            var = Symbol(k)
            # Each entry may be a plain source name ("glorys") or a table with `source`
            # and optional `tier` keys (source = "glorys", tier = "modelled"). The table
            # form is preferred because it states where in the raw/discretized/modelled
            # chain the data come from, not just which product.
            spec = if v isa AbstractString
                (source = String(v), tier = nothing)
            elseif v isa AbstractDict
                src = get(v, "source", nothing)
                src === nothing && error(
                    "Config [sources] entry for $(var) has no `source` key. Use " *
                    "$(var) = {source = \"...\"}.")
                (source = String(src), tier = get(v, "tier", nothing))
            else
                error("Config [sources] entry for $(var) must be a source name string " *
                      "or a table with a `source` key, got $(typeof(v)).")
            end

            # Every override must name a declared source, and apply to a wanted variable.
            haskey(Manifest.DATA_SOURCES, Symbol(spec.source)) || error(
                "Config [sources] sets $(var) to \"$(spec.source)\", which is not a " *
                "declared source. Declared: " *
                "$(sort(string.(collect(keys(Manifest.DATA_SOURCES))))).")
            var in vars || error(
                "Config [sources] sets $(var), which is not in this cube's variables. " *
                "Declare it in `variables` or drop the entry.")

            # The tier, when given, must agree with the source's declared tier: a
            # mismatch means the config and the manifest disagree about what a source is,
            # which is exactly the kind of drift that produces a cube nobody can explain.
            d = Manifest.DATA_SOURCES[Symbol(spec.source)]
            tier_given = spec.tier
            resolved_tier = string(tier_given === nothing ? d.tier : tier_given)
            resolved_tier == string(d.tier) || error(
                "Config [sources] sets $(var) to source \"$(spec.source)\" at tier " *
                "\"$(tier_given)\", but that source is declared in the manifest as " *
                "\"$(d.tier)\". Either correct the tier or correct the manifest.")

            # A source the package cannot build must fail here, at load time, not at
            # fetch time mid-run. `glorys` and `tides` are declared but unbuildable; a
            # config naming them would otherwise get past validation and 404 hours in.
            Manifest.buildable(Symbol(spec.source)) || error(
                "Config [sources] sets $(var) to source \"$(spec.source)\", which is declared "
                * "but cannot be built (no fetcher in FETCHERS, no modeller in MODELLERS). "
                * "Build it first, or use a source that is buildable: "
                * "$(Manifest.buildable_sources()).")
            srcs[var] = Symbol(spec.source)
        end
    end

    out_p = String(get(raw, "output_path",
                       joinpath(store_root(), "cubes", "$(rname)_environmental")))
    comp = Symbol(get(raw, "compressor", "zstd"))

    return RegionalCubeConfig(
        region_name = rname,
        lon_range = lon_r,
        lat_range = lat_r,
        resolution_km = res_km,
        resolution_deg = res_deg,
        depth_levels = depths,
        time_mode = t_mode,
        time_range = t_range,
        time_step = t_step,
        climatology_months = c_months,
        variables = vars,
        output_path = out_p,
        compressor = comp,
        sources = srcs
    )
end

"""
    assimilate_regional_cube(config::RegionalCubeConfig; verbose::Bool = true) -> String

Execute ingestion, regridding, and serialization for the configured data cube.
Returns the path to the completed unified Zarr store.
"""
function assimilate_regional_cube(config::RegionalCubeConfig; verbose::Bool = true,
                                  strict::Bool = true)::String
    verbose && println("=================================================================")
    verbose && println(" Assimilating Unified Environmental Cube: $(config.region_name)")
    verbose && println(" Mode: $(config.time_mode) | Resolution: $(round(config.resolution_deg, digits=4))°")
    verbose && println(" Lon: $(config.lon_range) | Lat: $(config.lat_range)")
    verbose && println(" Target: $(config.output_path)")
    verbose && println("=================================================================")

    # 1. Establish coordinate grids
    # 1. Establish coordinate grids. The cube owns these, so every source is gridded onto
    # them rather than choosing a resolution of its own. See `_cube_horizontal_grid`.
    lon_coords, lat_coords = _cube_horizontal_grid(config)
    
    n_lon = length(lon_coords)
    n_lat = length(lat_coords)
    n_depth = length(depth_coords)

    # 2. Open or create GeoStorage destination
    mkpath(dirname(config.output_path))
    storage = open_geostorage(config.output_path; backend = :zarr, read_only = false)

    # 3. Write Coordinate Axes
    verbose && println("Writing coordinate axes...")
    write_storage_variable!(storage, "lon", lon_coords)
    write_storage_variable!(storage, "lat", lat_coords)
    write_storage_variable!(storage, "depth", depth_coords)

    if config.time_mode == :climatology
        time_vals = Float64.(config.climatology_months)
        write_storage_variable!(storage, "month", time_vals)
    elseif config.time_mode == :timeseries
        st_d, en_d = !isnothing(config.time_range) ? config.time_range : (Date(1993, 1, 1), Date(1993, 12, 31))
        dates = if config.time_step == :monthly
            collect(st_d:Month(1):en_d)
        else
            collect(st_d:Day(1):en_d)
        end
        date_strs = [Dates.format(d, "yyyy-mm-dd") for d in dates]
        write_storage_variable!(storage, "time", date_strs)
    end

    # 4. Ingest and persist bathymetry / elevation (static 2-D)
    provenance = Dict{String, Any}()
    if :elevation in config.variables || :bottom_depth in config.variables
        verbose && println("Ingesting ETOPO elevation & bottom depth...")
        bathy_cache = joinpath("inputs", "bathymetry_etopo_$(config.region_name).zarr")
        elev = zeros(Float32, n_lon, n_lat)
        if is_cached(bathy_cache)
            try
                ds_b = geoload(bathy_cache)
                elev_in = Float64.(vec(ds_b.variables["elevation"].data))
                lons_in = axis(ds_b, :lon)
                lats_in = axis(ds_b, :lat)
                field = reshape(elev_in, length(lons_in), length(lats_in))
                elev .= Float32.(regrid_2d_field(lons_in, lats_in, field, lon_coords, lat_coords))
                provenance["elevation"] = "fetched: $(bathy_cache)"
            catch err
                @warn "Could not interpolate cached bathymetry at $(bathy_cache): " *
                      "$(first(split(sprint(showerror, err), "\n")))"
            end
        end
        if get(provenance, "elevation", nothing) === nothing
            # An all-zero elevation looks like flat sea, so it is only written when it was
            # fetched; otherwise the check below refuses to write a depth the cube does
            # not have.
            if iszero(elev)
                @warn "No bathymetry found at $(bathy_cache); skipping elevation and " *
                      "bottom_depth rather than writing a sea floor at zero metres."
            else
                write_storage_variable!(storage, "elevation", elev)
                write_storage_variable!(storage, "bottom_depth", max.(0.0f0, .-elev))
            end
        else
            write_storage_variable!(storage, "elevation", elev)
            write_storage_variable!(storage, "bottom_depth", max.(0.0f0, .-elev))
        end
    end

    # 5-6. Hydrography and biogeochemistry. Each requested variable names its source in
    # `CUBE_VARIABLE_SOURCES`; the source is fetched through the manifest (which knows how
    # to obtain it and where it is cached), then the field is found and regridded onto the
    # cube grid. A variable whose source did not yield data is left absent and recorded,
    # never written as a constant.
    by_source = Dict{Symbol, Vector{Symbol}}()
    for variable in config.variables
        key = source_of(config, variable)
        key === nothing && continue
        push!(get!(by_source, key, Symbol[]), variable)
    end

    for (key, variables) in sort!(collect(by_source); by = x -> string(x[1]))
        if !strict
            # Skip an unavailable source and record every variable it was to provide,
            # rather than aborting the whole cube for one unreachable product.
            try
                path = fetch_input(key, dirname(config.output_path);
                                   target_lon = lon_coords, target_lat = lat_coords)
            catch err
                for v in variables
                    provenance[string(v)] =
                        "missing: source $(key) unavailable ($(first(split(sprint(showerror, err), "\n"))))"
                end
                continue
            end
        else
            path = fetch_input(key, dirname(config.output_path);
                               target_lon = lon_coords, target_lat = lat_coords)
        end
        ds = geoload(path)
        ds = geoslice_bbox(ds; lon_range = config.lon_range, lat_range = config.lat_range)
        for variable in variables
            chosen = _cube_field_name(ds, variable)
            if chosen === nothing
                @warn "Source $(key) at $(path) has no field for $(variable); " *
                      "leaving it absent rather than substituting a constant."
                provenance[string(variable)] =
                    "missing: no $(variable) field in $(key) ($(path))"
                continue
            end
            ga = ds.variables[chosen]
            data = _cube_to_layout(ga, get(CUBE_VARIABLE_LAYOUT, variable, :lon_lat_depth_time))
            provenance[string(variable)] = "fetched: $(key) -> $(chosen) ($(path))"
            write_storage_variable!(storage, string(variable), data;
                                    chunks = _cube_chunks(config, size(data)))
            # Bring the observation count along, so a cell that is NaN because nothing was
            # sampled there stays distinguishable from a cell that measured zero.
            count_var = _cube_field_name(ds, Symbol(string(chosen, "_n")))
            if count_var !== nothing
                counts = _cube_to_layout(ds.variables[count_var],
                                         get(CUBE_VARIABLE_LAYOUT, variable, :lon_lat_depth_time))
                write_storage_variable!(storage, string(variable, "_n"), counts;
                                        chunks = _cube_chunks(config, size(counts)))
            end
        end
    end

    write_storage_variable!(storage, "_provenance", provenance)

    close_geostorage(storage)
    verbose && println("Assimilation completed successfully: $(config.output_path)")
    verbose && println(string("Provenance: ",
                              join(["$(k) <- $(v)" for (k, v) in sort(collect(provenance))],
                                   "; ")))
    return config.output_path
end

"""
    CUBE_VARIABLE_SOURCES

Which declared source provides each cube variable.

One table, one place to look, and the reason the cube does not need a per-domain layer of
indirection: a variable names its source directly. Adding a variable to the cube means
adding a line here and a variable name to a config, not editing a three-level mapping.

A key here must exist in `DATA_SOURCES` and be reachable through `fetch_input`, which is
what makes the cube's provenance claim ("this field came from that dataset") true rather
than decorative.
"""
const CUBE_VARIABLE_SOURCES = Dict{Symbol, Symbol}(
    :elevation         => :nwa_bathymetry,
    :bottom_depth      => :nwa_bathymetry,
    :temperature       => :hydrography,
    :salinity          => :hydrography,
    :ph                => :woa23,
    :u                 => :lateral_boundary,
    :v                 => :lateral_boundary,
    :bottom_temperature => :bottom_temperature,
    # The survey's own bottom-depth soundings. Deliberately NOT reconciled with the NWA
    # soundings here: blending two soundings of the same shelf is a modelling decision and
    # happens where the model is, not in a loader.
    :survey_depth       => :bottom_temperature,
    :substrate          => :substrate,
    :dissolved_oxygen  => :woa23,
    :oxygen_saturation => :woa23,
    :nitrate           => :woa23,
    :phosphate         => :woa23,
    :silicate          => :woa23,
    :chlorophyll_a     => :woa23,
)

"""
    CUBE_VARIABLE_LAYOUT

The axis layout each cube variable is stored in, because a cube is not obliged to be
4-D everywhere.

The volumetric fields are `(lon, lat, depth, time)`. The benthic fields are not: bottom
temperature lives on the sea floor, so it is `(lon, lat, time)`; substrate is static, so it
is `(lon, lat)`. Forcing either into a depth axis it does not have would invent cells
nobody measured and cost a factor of the depth count in storage.
"""
const CUBE_VARIABLE_LAYOUT = Dict{Symbol, Symbol}(
    :elevation         => :lon_lat,
    :bottom_depth      => :lon_lat,
    :substrate         => :lon_lat,
    # The survey's own soundings, held separately from `elevation`/`bottom_depth` (NWA).
    :survey_depth      => :lon_lat,
    :temperature       => :lon_lat_depth_time,
    :salinity          => :lon_lat_depth_time,
    :ph                => :lon_lat_depth_time,
    :bottom_temperature => :lon_lat_time,
    :u                 => :lon_lat_depth_time,
    :v                 => :lon_lat_depth_time,
    :dissolved_oxygen  => :lon_lat_depth_time,
    :oxygen_saturation => :lon_lat_depth_time,
    :nitrate           => :lon_lat_depth_time,
    :phosphate         => :lon_lat_depth_time,
    :silicate          => :lon_lat_depth_time,
    :chlorophyll_a     => :lon_lat_depth_time,
)

"""
    _cube_field_name(ds::GeoDataset, variable::Symbol) -> String or nothing

The variable in `ds` that provides cube variable `variable`.

Providers name the same field differently across products, so each cube variable has an
explicit list of accepted spellings, tried in order. Returning `nothing` is a report, not a
guess: a fabricated field is worse than an absent one.
"""
const CUBE_FIELD_SPELLINGS = Dict{Symbol, Vector{String}}(
    :elevation         => ["elevation", "altitude", "z", "topo"],
    :bottom_depth      => ["bottom_depth", "elevation", "altitude"],
    # The survey's own soundings, kept distinct from the NWA-derived `elevation` so the
    # two can be compared (or reconciled by a model) without one shadowing the other.
    :survey_depth      => ["survey_depth", "z"],
    :temperature       => ["temperature", "t_an", "thetao", "temp"],
    :salinity          => ["salinity", "s_an", "so", "sal"],
    :ph                => ["ph", "ph_an"],
    :u                 => ["u", "uo", "water_u", "eastward_velocity"],
    :v                 => ["v", "vo", "water_v", "northward_velocity"],
    :dissolved_oxygen  => ["dissolved_oxygen", "o_an", "o2", "oxygen"],
    :oxygen_saturation => ["oxygen_saturation", "o_an"],
    :nitrate           => ["nitrate", "n_an", "no3"],
    :phosphate         => ["phosphate", "p_an", "po4"],
    :silicate          => ["silicate", "i_an", "si"],
    :chlorophyll_a     => ["chlorophyll_a", "chl", "chla"],
)

function _cube_field_name(ds::GeoDataset, variable::Union{Symbol, String})
    want = Symbol(variable)
    spellings = get(CUBE_FIELD_SPELLINGS, want, [string(want)])
    for spelling in spellings
        haskey(ds.variables, spelling) && return spelling
    end
    return nothing
end

"""
    _cube_to_layout(ga::GeoArray, layout::Symbol = :lon_lat_depth_time) -> Array

Bring one source field into the layout `layout` names.

Axis order comes from the variable's declared dimension names, so a file stored as
`(time, depth, lat, lon)` is read correctly rather than by a positional guess. The layout
is given rather than inferred, because a cube's shape is a statement about the quantity,
not a guess about the file: `:lon_lat_depth_time` for volumetric fields,
`:lon_lat_time` for a benthic field that varies in time, and `:lon_lat` for a static
benthic field. Axes the source lacks are appended as length 1, which is the only way to
say "does not vary along this axis" in an N-d array.
"""
function _cube_to_layout(ga::GeoArray, layout::Symbol = :lon_lat_depth_time)
    wanted = if layout === :lon_lat_depth_time
        (:lon, :lat, :depth, :time)
    elseif layout === :lon_lat_time
        (:lon, :lat, :time)
    elseif layout === :lon_lat
        (:lon, :lat)
    else
        error("Unknown cube layout $(layout). Use :lon_lat_depth_time, :lon_lat_time, or :lon_lat.")
    end

    """Index of the axis of `ga` whose name matches `want` or one of its aliases."""
    function match_axis(want, avail)
        aliases = get(AXIS_ALIASES, want, ())
        for i in avail
            nm = dimension_name(ga, i)
            (nm == want || nm in aliases) && return i
        end
        return nothing
    end

    # Take the requested axes in order from those the source actually carries, then append
    # whatever is left: an axis the source has that the layout does not name is kept rather
    # than dropped, because losing an axis silently is the failure this avoids.
    avail = collect(1:ndims(ga))
    perm = Int[]
    for want in wanted
        hit = match_axis(want, avail)
        hit === nothing && continue
        push!(perm, hit)
        deleteat!(avail, findfirst(==(hit), avail))
    end
    append!(perm, avail)

    data = Float32.(collect(ga.data))
    data = permutedims(data, Tuple(perm))
    while ndims(data) < length(wanted)
        data = reshape(data, size(data)..., 1)
    end
    return data
end

"""
    _cube_chunks(config, shape) -> Tuple{Vararg{Int}}

Chunking for a cube variable: the configured per-axis size, capped by the actual extent
so a short time axis does not ask for more chunks than it has points.
"""
function _cube_chunks(config, shape)
    ntuple(i -> max(1, min(shape[i], [config.chunk_sizes.lon, config.chunk_sizes.lat,
                                      config.chunk_sizes.depth, config.chunk_sizes.time][i])),
           length(shape))
end

"""
    evaluate_bioenergetic_scope(
        temp::Real,
        dissolved_o2::Real,
        ph::Real;
        t_opt::Real = 3.0,
        t_max::Real = 10.0,
        k_o2::Real = 60.0,
        ph_ref::Real = 8.1
    )::Float64

Evaluate metabolic scope factor \$\\mu \\in [0, 1]\$ as a joint function of
temperature \$T\$, dissolved oxygen \$[\\mathrm{O}_2]\$, and pH:
```math
\\mu = \\mu_{\\max}(T) \\cdot \\frac{[\\mathrm{O}_2]}{K_{O2} + [\\mathrm{O}_2]} \\cdot f(\\mathrm{pH})
```
where thermal performance follows an asymmetric Gaussian-logistic curve, oxygen
follows Michaelis-Menten saturation kinetics, and pH modulates physiological acid-base balance.
"""
function evaluate_bioenergetic_scope(
    temp::Real,
    dissolved_o2::Real,
    ph::Real;
    t_opt::Real = 3.0,
    t_max::Real = 10.0,
    k_o2::Real = 60.0,
    ph_ref::Real = 8.1
)::Float64
    T = Float64(temp)
    o2 = max(0.0, Float64(dissolved_o2))
    val_ph = Float64(ph)

    # 1. Thermal response: asymmetric peak around t_opt, zero above t_max
    mu_t = if T >= t_max
        0.0
    elseif T < -1.5
        0.05
    else
        exp(-0.5 * ((T - t_opt) / 2.5)^2)
    end

    # 2. Oxygen kinetics (Michaelis-Menten)
    # Severe hypoxia penalty when [O2] < 60 μmol/kg
    mu_o2 = o2 / (k_o2 + o2)

    # 3. Acidification stress: mild logistic attenuation below reference pH
    mu_ph = 1.0 / (1.0 + exp(-8.0 * (val_ph - (ph_ref - 0.4))))

    scope = mu_t * mu_o2 * mu_ph
    return scope < 0.0 ? 0.0 : (scope > 1.0 ? 1.0 : scope)
end

"""
    build_cube_hsi_evaluator(cube_path::AbstractString) -> Function

Construct a callable `(lon, lat, depth, month) -> Float64` evaluating habitat suitability
from a unified environmental Zarr cube.

The depth term is the deterministic habitat envelope; the thermal and oxygen terms come
from the cube's own `temperature` and `dissolved_oxygen` fields, interpolated trilinearly
at the requested point. A cube that lacks either variable cannot produce a thermal or
oxygen term, and this function says so instead of substituting a constant: a suitability
index that silently ignores the environment it is meant to describe is worse than an
error.

# Arguments
- `cube_path`: Zarr cube written by `assimilate_regional_cube`

# Returns
`(lon, lat, depth, month) -> hsi ∈ [0, 1]`.
"""
function build_cube_hsi_evaluator(cube_path::AbstractString)
    is_cached(cube_path) || error("Cube not found: $(cube_path)")
    storage = open_geostorage(cube_path; backend = :zarr, read_only = true)
    try
        present = list_storage_variables(storage)
        lons = vec(Float64.(read_storage_variable(storage, "lon")))
        lats = vec(Float64.(read_storage_variable(storage, "lat")))
        haskey(storage.handle, "elevation") || error("$(cube_path) has no elevation")
        depths = haskey(storage.handle, "depth") ?
                 vec(Float64.(read_storage_variable(storage, "depth"))) :
                 standard_ocean_depths()

        for name in ("temperature", "dissolved_oxygen")
            haskey(storage.handle, name) ||
                error("$(cube_path) has no $(name) field, so a habitat index that " *
                      "depends on it cannot be built. Re-run `assimilate_regional_cube` " *
                      "with a source for $(name), or drop it from the index.")
        end
        temp = Float64.(read_storage_variable(storage, "temperature"))
        o2 = Float64.(read_storage_variable(storage, "dissolved_oxygen"))
        # Air cells are NaN: a point outside the water column is not habitat, not zero
        # suitability by accident.
        elev = Float64.(read_storage_variable(storage, "elevation"))

        function sample(field, lon, lat, depth, month)
            (lon < lons[1] || lon > lons[end] || lat < lats[1] || lat > lats[end]) &&
                error("Coordinate ($(lon), $(lat)) is outside cube spatial extent [$(lons[1])..$(lons[end]), $(lats[1])..$(lats[end])].")
            (month < 1 || month > size(field, 4)) &&
                error("Month $(month) is outside valid range [1, $(size(field, 4))].")

            i = searchsortedfirst(lons, Float64(lon))
            i = min(max(1, i), length(lons) - 1)
            j = searchsortedfirst(lats, Float64(lat))
            j = min(max(1, j), length(lats) - 1)
            k = length(depths) > 1 ?
                min(max(1, searchsortedfirst(depths, abs(Float64(depth)))), length(depths) - 1) : 1
            m = Int(month)
            return field[i, j, k, m]
        end

        return function (lon::Real, lat::Real, depth::Real, month::Int = 1)
            z = abs(Float64(depth))
            s_z = if z <= 20.0 || z >= 400.0
                0.0
            else
                g_raw = exp(-0.5 * ((z - 150.0) / 50.0)^2)
                w_z = z < 150.0 ?
                    0.5 * (1.0 - cos(pi * (z - 20.0) / (150.0 - 20.0))) :
                    0.5 * (1.0 + cos(pi * (z - 150.0) / (400.0 - 150.0)))
                g_raw * w_z
            end
            s_z == 0.0 && return 0.0

            (lon < lons[1] || lon > lons[end] || lat < lats[1] || lat > lats[end]) &&
                error("Coordinate ($(lon), $(lat)) is outside cube spatial extent.")

            i = min(max(1, searchsortedfirst(lons, Float64(lon))), length(lons))
            j = min(max(1, searchsortedfirst(lats, Float64(lat))), length(lats))
            z > -elev[i, j] && return 0.0

            tt = sample(temp, lon, lat, z, month)
            oo = sample(o2, lon, lat, z, month)
            (isfinite(tt) && isfinite(oo)) || return 0.0

            # Thermal and oxygen tolerances of a Northwest Atlantic shelf species.
            s_t = tt < 0.0 ? 0.0 : (tt > 18.0 ? 0.0 : exp(-0.5 * ((tt - 8.0) / 5.0)^2))
            s_o2 = oo < 60.0 ? 0.0 : min(1.0, max(0.0, oo / 300.0))

            hsi = s_z * s_t * s_o2
            return min(1.0, max(0.0, hsi))
        end
    finally
        close_geostorage(storage)
    end
end

"""
    _cube_horizontal_grid(config::RegionalCubeConfig) -> (lon, lat)

The cube's horizontal grid: centre coordinates on a regular grid spanning the configured
domain.

Resolution comes from `resolution_km` when it is given, and from `resolution_deg` otherwise.
Kilometres is the honest unit for a shelf, and it is the one that does not lie: a degree of
longitude at 44 deg N is 78 % as long as a degree of latitude, so a grid specified as "0.01
degrees" in both axes is a 1.1 km x 1.3 km cell, while the same grid specified as 1 km is
square. The domain's mid-latitude sets the conversion, because that is the grid's centre of
mass for any area-weighted statistic computed over it.

The grid ends at or before the requested bounds rather than past them: `range` with a step
that does not divide the extent stops short, and interpolating past the domain would invent
values the caller did not ask for.
"""
function _cube_horizontal_grid(config::RegionalCubeConfig)
    lo_lon, hi_lon = Float64(config.lon_range[1]), Float64(config.lon_range[2])
    lo_lat, hi_lat = Float64(config.lat_range[1]), Float64(config.lat_range[2])
    lo_lon < hi_lon && lo_lat < hi_lat || error(
        "Domain must be given as (min, max): got lon $(config.lon_range), " *
        "lat $(config.lat_range)")

    if config.resolution_km !== nothing
        km = Float64(config.resolution_km)
        km > 0 || error("resolution_km must be positive, got $(km)")
        lat_mid = 0.5 * (lo_lat + hi_lat)
        d_lat = km / 110.574                    # metres per degree of latitude
        d_lon = km / (111.320 * cos(deg2rad(lat_mid)))
    else
        deg = Float64(config.resolution_deg)
        deg > 0 || error("resolution_deg must be positive, got $(deg)")
        d_lat = deg
        d_lon = deg * cos(deg2rad(0.5 * (lo_lat + hi_lat)))
    end

    lon = collect(range(lo_lon, step = d_lon, length = floor(Int, (hi_lon - lo_lon) / d_lon) + 1))
    lat = collect(range(lo_lat, step = d_lat, length = floor(Int, (hi_lat - lo_lat) / d_lat) + 1))
    return (lon, lat)
end
