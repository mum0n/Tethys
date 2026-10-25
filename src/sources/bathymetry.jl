# ==============================================================================
# bathymetry.jl -- Bathymetry fetch, load, save, and query operations
# Supports ERDDAP (NOAA), ETOPO2022, and generic NetCDF/Zarr formats.
# ==============================================================================

using NCDatasets
using Random
using Statistics

# Optional NumericalEarth for ETOPO bathymetry
const _has_numerical_earth = try
    import NumericalEarth
    true
catch
    false
end

"""
    fetch_erddap_bathymetry(;
        dataset_id::AbstractString = "ETOPO_2022_v1_15s",
        lon_range::Tuple{Real, Real} = (-71.0, -53.0),
        lat_range::Tuple{Real, Real} = (40.0, 48.5),
        stride::Int = 1,
        output_path::AbstractString = joinpath("inputs", "bathymetry.zarr"),
        backend::Symbol = :zarr,
        verbose::Bool = true
    ) -> GeoDataset

Fetch bathymetry from NOAA ERDDAP and save as a GeoDataset.

# Arguments
- `dataset_id`: ERDDAP dataset ID (e.g., "ETOPO_2022_v1_15s", "etopo180", "srtm30plus")
- `lon_range`: Longitude bounds (min, max) in degrees east
- `lat_range`: Latitude bounds (min, max) in degrees north
- `stride`: Subsampling stride (1 = full resolution)
- `output_path`: Destination path (Zarr directory or NetCDF file)
- `backend`: Storage backend (:zarr, :ncdatasets)
- `verbose`: Print progress messages

# Returns
GeoDataset with elevation, lon, lat coordinates.
"""
function fetch_erddap_bathymetry(;
    dataset_id::AbstractString = "ETOPO_2022_v1_15s",
    lon_range::Tuple{Real, Real} = (-71.0, -53.0),
    lat_range::Tuple{Real, Real} = (40.0, 48.5),
    stride::Int = 1,
    output_path::AbstractString = joinpath("inputs", "bathymetry.zarr"),
    backend::Symbol = :zarr,
    force::Bool = false,
    verbose::Bool = true
)
    # Reuse the cached store: the manifest already resolved the destination, and
    # re-downloading a gridded product to rebuild an identical store is wasted work.
    if !force && is_cached(output_path)
        verbose && println("Using cached bathymetry at $(output_path)")
        return geoload(output_path; backend = backend)
    end

    erddap_vars = Dict(
        "ETOPO_2022_v1_15s" => "z",
        "etopo180"          => "altitude",
        "srtm30plus"        => "elevation"
    )
    var = get(erddap_vars, dataset_id, "z")

    # ERDDAP subset syntax uses literal brackets; encode them for a safe URL. The bounds are
    # ordered here rather than assumed from the caller: `lon_range` is a range, and handing
    # ERDDAP an inverted one returns an empty subset that loads without complaint.
    min_lon, max_lon = minmax(lon_range...)
    min_lat, max_lat = minmax(lat_range...)
    # Coordinate-value bounds are parenthesised at BOTH ends: the earlier code parenthesised
    # only the start -- `z[(45.0):1:46.9]` -- which ERDDAP answers with a 400. The dataset's
    # own `defaultGraphQuery` is `z[(start):(stop)][(start):(stop)]`, so mirror that, and
    # emit an explicit stride only when one is asked for.
    lat_rng = stride <= 1 ? "($(min_lat)):($(max_lat))" : "($(min_lat)):$(stride):($(max_lat))"
    lon_rng = stride <= 1 ? "($(min_lon)):($(max_lon))" : "($(min_lon)):$(stride):($(max_lon))"
    url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/$(dataset_id).nc?" *
          "$(var)%5B$(lat_rng)%5D%5B$(lon_rng)%5D"

    verbose && println("Fetching bathymetry from ERDDAP...")
    # Scratch beside the destination on purpose: a large ERDDAP file must not land in the OS temp dir.
    scratch = joinpath(dirname(output_path), ".bathymetry-$(dataset_id).nc")
    download_to_path(url, scratch; verbose = verbose, retries = 2)
    # Load the downloaded NetCDF and convert to GeoDataset
    ds = geoload(scratch; backend=:ncdatasets)
    # Save as Zarr (or other backend)
    geosave(output_path, ds; backend=backend, overwrite = force)
    rm(scratch, force=true)
    verbose && println("Saved bathymetry to $(output_path)")
    return ds
end

"""
    etopo_bathymetry_field(;
        lon_range::Tuple{Real, Real} = (-71.0, -53.0),
        lat_range::Tuple{Real, Real} = (40.0, 48.5),
        resolution::Real = 600,
        minimum_depth::Real = 0.0,
        interpolation_passes::Integer = 1,
        major_basins::Real = 1,
            cache = true  # Bool
    ) -> Matrix{Float64}

Get regional bathymetry as a raw matrix from ETOPO2022 using NumericalEarth.jl.
"""
function etopo_bathymetry_field(;
    lon_range::Tuple{Real, Real} = (-71.0, -53.0),
    lat_range::Tuple{Real, Real} = (40.0, 48.5),
    resolution::Real = 600,
    minimum_depth::Real = 0.0,
    interpolation_passes::Integer = 1,
    major_basins::Real = 1,
        cache = true  # Bool
)
    _has_numerical_earth || error("NumericalEarth.jl required for ETOPO bathymetry regridding. Add NumericalEarth to your environment.")
    
    resolution <= 0.0 && error("Resolution must be positive, got $(resolution) metres.")
    raw_n_lon = round(Int, abs(lon_range[2] - lon_range[1]) * 111_320.0 * cosd(lat_mid) / resolution)
    raw_n_lat = round(Int, abs(lat_range[2] - lat_range[1]) * 111_320.0 / resolution)
    n_lon = max(16, min(raw_n_lon, 21600))
    n_lat = max(16, min(raw_n_lat, 10800))
    
    scratch = NumericalEarth.LatitudeLongitudeGrid(
        NumericalEarth.CPU(), Float32;
        n_lon = n_lon, n_lat = n_lat,
        lon_min = lon_range[1], lon_max = lon_range[2],
        lat_min = lat_range[1], lat_max = lat_range[2]
    )
    
    etopo = NumericalEarth.ETOPO()
    bathy = NumericalEarth.regrid_bathymetry(etopo, scratch; major_basins = 1, interpolation_passes = 1)
    
    return Float64.(bathy)
end

"""
    load_bathymetry(filepath::AbstractString; backend=nothing) -> GeoDataset

Load bathymetry from a Tethys-compatible file (Zarr, NetCDF, GeoParquet, etc.),
auto-detecting the backend from the file extension when not specified.
"""
function load_bathymetry(filepath::AbstractString; backend = nothing)
    return load_dataset(filepath; backend = backend)
end

const load_bathymetry_geodata = load_bathymetry

"""
    save_bathymetry(ds::GeoDataset, filepath::AbstractString; backend=nothing) -> String

Save bathymetry GeoDataset to a file, auto-detecting the backend from the file extension
when not specified.
"""
function save_bathymetry(ds::GeoDataset, filepath::AbstractString; backend = nothing)
    mkpath(dirname(filepath))
    save_dataset(filepath, ds; backend = backend)
    return filepath
end

const save_bathymetry_geodata = save_bathymetry

function regrid_bathymetry_from_etopo(; lon_range::Tuple{Real, Real} = (-71.0, -53.0),
    lat_range::Tuple{Real, Real} = (40.0, 48.5),
    resolution::Real = 600,
    minimum_depth::Real = 0.0,
    interpolation_passes::Integer = 1,
    major_basins::Real = 1,
    cache::Bool = true,
)
    ds = etopo_bathymetry_field(; lon_range = lon_range, lat_range = lat_range,
                                resolution = resolution, minimum_depth = minimum_depth,
                                interpolation_passes = interpolation_passes,
                                major_basins = major_basins, cache = cache)
    lon_coords = Float64.(ds.axes[1])
    lat_coords = Float64.(ds.axes[2])
    elev = Float64.(ds.data)
    bat = CoordinateSystem(crs = "EPSG:4326")
    dims = (
        Dimension(name = :lon, size = length(lon_coords), coords = lon_coords, units = "degrees_east", dim_type = DIM_SPATIAL),
        Dimension(name = :lat, size = length(lat_coords), coords = lat_coords, units = "degrees_north", dim_type = DIM_SPATIAL)
    )
    vars = Dict(
        "elevation" => GeoArray(elev, dims, bat, Dict{String, Any}("units" => "m")),
    )
    return GeoDataset(variables = vars, coords = Dict(), dims = Dict(dims...), crs = bat, attrs = Dict{String, Any}(), backend = nothing, source = "ETOPO2022")
end

"""
    _etopo_from_store_tile(; lon_range, lat_range, verbose) -> GeoDataset

ETOPO read from the Tethys store — NumericalEarth's cached 15/60-arc-second relief tiles —
subset to the domain and turned into a `(lon, lat)` `elevation` dataset. Reading the store's
tile avoids re-downloading the whole relief model each run; the tile that spans the domain is
chosen by parsing the hemisphere/quadrant from its filename. `lons`/`lats` come straight from
the tile, `z` is the hyperslab over the domain.

This supersedes both the ERDDAP URL fetch (whole-planet download each run) and the
`NumericalEarth.regrid_bathymetry` shortcut, which built a grid with symbols NumericalEarth
does not export.
"""
function _etopo_from_store_tile(; lon_range::Tuple{Real, Real},
                                lat_range::Tuple{Real, Real},
                                verbose::Bool = true)
    min_lon, max_lon = minmax(lon_range...)
    min_lat, max_lat = minmax(lat_range...)

    # Search the central Tethys store first (store_root()/etopo), then NumericalEarth's
    # scratch space and any per-run inputs/ETOPO. The central store wins so an operator who
    # has relocated the registry has it honoured.
    candidate_dirs = AbstractString[store_path("etopo")]
    for base in (joinpath(homedir(), ".julia", "scratchspaces"),
                 joinpath(@__DIR__, "..", "..", "inputs"))
        isdir(base) && for proj in readdir(base; join = true)
            push!(candidate_dirs, joinpath(proj, "ETOPO"))
        end
    end
    tiles = String[]
    for base in candidate_dirs
        isdir(base) && for f in readdir(base; join = true)
            occursin("ETOPO", f) && endswith(f, ".nc") && push!(tiles, f)
        end
    end

    # A tile's name carries its coverage: N/S (latitude half) and E/W (longitude quadrant),
    # e.g. `ETOPO_2022_v1_60s_N90W180_surface.nc` is lat 0..90, lon -180..0.
    function tile_coverage(fname::AbstractString)
        m = match(r"_([NS])(\d+)([EW])(\d+)_surface\.nc$", fname)
        m === nothing && return nothing
        ns, nd, ew, ed = m[1], parse(Int, m[2]), m[3], parse(Int, m[4])
        lat = ns == "N" ? (0.0, Float64(nd)) : (-Float64(nd), 0.0)
        lon = ew == "W" ? (-Float64(ed), 0.0) : (0.0, Float64(ed))
        return (lat = lat, lon = lon)
    end

    tile = nothing
    for f in tiles
        c = tile_coverage(basename(f))
        c === nothing && continue
        if c.lat[1] <= min_lat && max_lat <= c.lat[2] && c.lon[1] <= min_lon && max_lon <= c.lon[2]
            tile = f
            break
        end
    end
    # The tiles carry GLOBAL coordinate arrays even when the filename names a quadrant, so
    # if no filename declares coverage, fall back to any available tile and let the arrays'
    # own range decide (a domain outside them yields an empty subset, which is an error).
    if tile === nothing && !isempty(tiles)
        tile = first(sort(tiles))
    end
    # Not in the store: return `nothing` so the caller can fall back to a GeoDataSources
    # download (store first, then fetch) rather than aborting the run.
    if tile === nothing
        verbose && println("ETOPO: no tile in the Tethys store covers this domain.")
        return nothing
    end
    verbose && println("Reading ETOPO from the store tile: $(basename(tile))")

    ds = Dataset(tile)
    lon = collect(Float64, ds["lon"][:])
    lat = collect(Float64, ds["lat"][:])
    li = findall(x -> min_lat <= x <= max_lat, lat)
    lj = findall(x -> min_lon <= x <= max_lon, lon)
    # The tile's first index is LONGITUDE: `z[lon, lat]`. Reading `z[lat, lon]` silently
    # transposes the whole relief model (a shelf read as the deep slope). Subset in (lon, lat)
    # order, keep it that way, and coerce the tile's `Union{Missing, Float32}` cells to a
    # concrete `Float64` (missing -> NaN), which is what a NetCDF writer can serialise.
    zsub = [v === missing ? NaN : Float64(v) for v in ds["z"][lj, li]]
    close(ds)

    lons = lon[lj]
    lats = lat[li]
    bat = CoordinateSystem(crs = "EPSG:4326")
    d_lon = Dimension(name = :lon, size = length(lons), coords = lons, units = "degrees_east", dim_type = DIM_SPATIAL)
    d_lat = Dimension(name = :lat, size = length(lats), coords = lats, units = "degrees_north", dim_type = DIM_SPATIAL)
    elev = zsub   # already (lon, lat)
    return GeoDataset(
        variables = Dict("elevation" => GeoArray(elev, (d_lon, d_lat), bat, Dict{String, Any}("units" => "m"))),
        coords = Dict{Symbol, GeoArray}(),
        dims = Dict(d_lon.name => d_lon, d_lat.name => d_lat),
        crs = bat,
        attrs = Dict{String, Any}("source" => "ETOPO2022 (store tile: $(basename(tile)))"),
        backend = nothing,
        source = "ETOPO2022")
end

"""
    bathymetry(; source::Symbol = :etopo, lon_range, lat_range, tier = nothing) -> GeoDataset

Fetch or load bathymetry for a domain by named source.

This is the dispatch point for "which bathymetry do you want" -> "how it is obtained". A
caller asks for a source and a range; which product that resolves to, what format it is
stored in, and whether it must be downloaded or already sits in the cache are decided here,
not by the caller.

Supported sources:

- `:etopo` -- ETOPO 2022, modelled and global. The failsafe: it exists everywhere, so a run
  that requests it cannot fail for want of data.
- `:nwa` -- measured Northwest Atlantic shelf soundings from the biodata record. Higher
  resolution than `:etopo` where soundings exist, and the preferred product for this shelf.
  It sits on a **different vertical datum**, so it is not interchangeable with `:etopo` and
  choosing it is a decision the caller configures, not something inferred here.

An unknown `source` errors rather than falling back to `:etopo`. A silent fallback would
hide a mistyped config value behind a plausible-looking bathymetry, and the run would
proceed on a different product from the one requested without anyone learning of it.
"""
function bathymetry(; source::Symbol = :etopo,
                    lon_range::Tuple{Real, Real},
                    lat_range::Tuple{Real, Real},
                    tier = nothing,
                    kwargs...)
    if source === :etopo
        return regrid_bathymetry_from_etopo(; lon_range = lon_range, lat_range = lat_range,
                                            kwargs...)
    elseif source === :nwa
        return fetch_nwa_bathymetry(; lon_range = lon_range, lat_range = lat_range,
                                    kwargs...)
    else
        known = isempty(keys(BATHYMETRY_SOURCES)) ? ":etopo, :nwa" :
                join(sort!(collect(keys(BATHYMETRY_SOURCES))), ", ")
        error("Unknown bathymetry source $(repr(source)). Known sources: $(known).")
    end
end

"""
    BATHYMETRY_SOURCE_ALIASES

How a bathymetry source may be spelled, mapped to the `BATHYMETRY_SOURCES` key it names.

A config is written by a person, and a person writes `"etopo2022"`, `"nwa"` or
`"scotian_shelf"` -- not the internal symbol. Every accepted spelling lands here once, so
an error about an unknown source can quote the spellings that *are* known rather than the
one symbol the dispatcher happens to use.

`"scotian_shelf"` names a **place**, not a product, and resolves to the product that is
better for that place: the CHS 15-arc-second soundings of the Northwest Atlantic shelf are
survey soundings of this shelf rather than a global model of it.
"""
const BATHYMETRY_SOURCE_ALIASES = Dict{String, Symbol}(
    # the global model: the failsafe, available everywhere
    "etopo"              => :etopo,
    "etopo2022"          => :etopo,
    "etopo_2022"         => :etopo,
    "global"             => :etopo,
    # the regional products
    "nwa"                => :nwa,
    "nwa_bathymetry"     => :nwa,
    "northwest_atlantic" => :nwa,
    "chs"                => :nwa,
    "chs15sec"           => :nwa,
    # places, resolved to the product that suits them
    "scotian_shelf"      => :nwa,
    "scotian"            => :nwa,
    "scotianshelf"       => :nwa,
    # "you decide, from where I am working"
    "auto"               => :auto,
    "best"               => :auto,
    "default"            => :auto,
    "region"             => :auto,
)

"""
    BATHYMETRY_COVERAGE

Where each regional bathymetry source actually has soundings: `source => (lon_range,
lat_range)` in degrees.

These are the bounds the manifest's own entry declares for the source. They are what makes
an automatic choice a statement about coverage rather than a guess: `:etopo` is a global
model and needs no bound, while `:nwa` is a regional survey and outside this box there are
no soundings to grid.
"""
const BATHYMETRY_COVERAGE = Dict{Symbol, NamedTuple}(
    :nwa => (lon_range = (-72.0, -42.0), lat_range = (40.0, 64.0)),
)

"""
    preferred_bathymetry_source(; source = nothing, lon_range = nothing,
                                lat_range = nothing) -> Symbol

Which bathymetry source to use, when the caller has no opinion of its own.

The order of decision matters, and each step is a fact rather than a preference:

1. **A named source wins.** The operator named it, and substituting a different product
   behind their back is exactly the silent substitution this package refuses. `"etopo2022"`
   gives ETOPO, `"scotian_shelf"` gives the NW Atlantic soundings, and an unrecognised
   spelling is an error listing what *is* accepted.
2. **Otherwise, coverage decides.** When the requested domain lies entirely inside a
   regional source's soundings, that source is used: on the Scotian Shelf the CHS
   soundings are survey measurements of the shelf being modelled, while ETOPO is a global
   model that happens to include it.
3. **Otherwise `:etopo`.** It is modelled and global, so it exists everywhere and a
   request for it cannot fail for want of data.

`source = nothing`, `""` or `"auto"` all mean "you decide" and take step 2. Pass
`lon_range`/`lat_range` as `nothing` to skip the coverage test and go straight to `:etopo`.
"""
function preferred_bathymetry_source(; source = nothing,
                                     lon_range = nothing,
                                     lat_range = nothing)
    requested = _bathymetry_source_key(source)
    requested === :auto || return requested

    # No stated preference: let coverage decide. A domain straddling the edge of a regional
    # survey is not covered by it, because the part outside would be silently replaced by
    # the global model and the result would be two products in one field.
    if lon_range !== nothing && lat_range !== nothing
        lon_lo, lon_hi = minmax(lon_range...)
        lat_lo, lat_hi = minmax(lat_range...)
        for key in sort!(collect(keys(BATHYMETRY_COVERAGE)); by = string)
            cov = BATHYMETRY_COVERAGE[key]
            inside = lon_lo >= cov.lon_range[1] && lon_hi <= cov.lon_range[2] &&
                     lat_lo >= cov.lat_range[1] && lat_hi <= cov.lat_range[2]
            inside && return key
        end
    end
    return :etopo
end

"""
    _bathymetry_source_key(source) -> Symbol

Normalise one spelling of a bathymetry source to its `BATHYMETRY_SOURCES` key, or to the
private `:auto` marker meaning "decide from where I am working".

An unrecognised spelling errors and quotes the accepted ones. Silently mapping it to
`:etopo` would let a typo run a whole simulation on the wrong sea floor.
"""
function _bathymetry_source_key(source)
    source === nothing && return :auto
    text = strip(lowercase(string(source)))
    (isempty(text) || text in ("auto", "best", "default", "region")) && return :auto
    key = get(BATHYMETRY_SOURCE_ALIASES, text, nothing)
    if key === nothing
        spellings = join(sort!(collect(keys(BATHYMETRY_SOURCE_ALIASES))), ", ")
        resolved_to = join(sort!(collect(keys(BATHYMETRY_SOURCES)); by = string), ", ")
        error("Unknown bathymetry source $(repr(source)). Accepted spellings: $(spellings). " *
              "These resolve to $(resolved_to).")
    end
    return key
end

"""
    _BATHYMETRY_MANIFEST_KEY

Which declared source in `DATA_SOURCES` each `BATHYMETRY_SOURCES` key corresponds to.

The two tables are keyed differently because they answer different questions: `bathymetry`
dispatches over products, while the manifest declares provenance. Without this mapping a
requested `tier` could not be checked against anything.
"""
const _BATHYMETRY_MANIFEST_KEY = Dict{Symbol, Symbol}(
    :etopo => :bathymetry,
    :nwa   => :nwa_bathymetry,
)

"""
    _check_bathymetry_tier(source, tier)

Refuse a `tier` that contradicts the tier `source` is declared at in the manifest.

A tier is where in the raw / discretized / modelled chain the data sit, and it is a property
of the source rather than of the request. A config asking for the NW Atlantic soundings at
`:raw` and the manifest declaring them `:discretized` are two statements about what a product
*is*, and proceeding would produce a field whose provenance nobody can state. A source with
nothing declared constrains nothing, so it passes.
"""
function _check_bathymetry_tier(source::Symbol, tier)
    tier === nothing && return nothing
    manifest_key = get(_BATHYMETRY_MANIFEST_KEY, source, nothing)
    manifest_key === nothing && return nothing
    declared = get(Manifest.DATA_SOURCES, manifest_key, nothing)
    declared === nothing && return nothing
    requested = Symbol(tier)
    requested == declared.tier || error(
        "Bathymetry source $(source) is declared at tier :$(declared.tier), but tier " *
        ":$(requested) was requested. Correct the tier, or correct the manifest -- the two " *
        "are statements about what this product is, and they must agree.")
    return nothing
end

"""
    bathymetry_artefact(source::Symbol) -> String

The file name a bathymetry source is written under: `bathymetry_etopo.nc` for the
modelled global surface, which is a NetCDF file, and `bathymetry_nwa.zarr` for the
measured sounding record, which is a Zarr group.

The name carries the format because the two sources write different kinds of artefact and
one path cannot hold both. A NetCDF file written where a Zarr group is expected is read
back as a directory and fails with a permission error; a Zarr group written where a file
is expected fails the same way. Naming the two sources apart is what stops one run's
product from becoming the next run's "file already exists".
"""
function bathymetry_artefact(source::Symbol)
    source === :nwa && return "bathymetry_nwa.zarr"
    source === :etopo && return "bathymetry_etopo.nc"
    declared = join(sort!(collect(keys(BATHYMETRY_SOURCES)); by = string), ", ")
    error("No bathymetry artefact is defined for source $(source). " *
          "Declared sources: $(declared).")
end

"""
    bathymetry_dataset(; source = nothing,
                       lon_range::Tuple{Real, Real},
                       lat_range::Tuple{Real, Real},
                       tier = nothing,
                       output_path::Union{Nothing, AbstractString} = nothing,
                       force::Bool = false,
                       verbose::Bool = true,
                       kwargs...) -> GeoDataset

Bathymetry for a domain as a `GeoDataset`, by named source and range, with an optional
durable artefact.

This is the run-path counterpart to `bathymetry`: the single place that answers "which
product, at which tier, obtained how", so a caller names a source and a range and learns
nothing about formats or files. `output_path` is what makes the resolution durable and
shared:

- when given and already cached, that artefact is loaded and no product is re-fetched, so
  every reader in a run sees the *same* sea floor;
- when given and absent, the resolved product is fetched and saved there;
- when absent, the product is returned in memory only.

`source` is passed through `preferred_bathymetry_source`, so `nothing`/`"auto"` choose by
coverage and `"scotian_shelf"` names the shelf. `tier` selects the processing level the
source is obtained at, and `force` re-fetches past a cache.

`verbose` is a print switch only: it never decides what is fetched.

`output_path` may name a directory, in which case the source's artefact goes inside it
under its canonical name (`bathymetry_artefact`), or a file path, which is used as given.
A directory is the usual case for a run: the run's inputs directory holds the products
this run read, each named for what it is. Without it the product is written under
`inputs/` in the working directory, which is a scratch location and not a durable home --
a caller that wants the product to outlive the process must name where it goes.

# Example
```julia
ds = bathymetry_dataset(; source = "scotian_shelf", lon_range = (-68.0, -57.0),
                        lat_range = (42.0, 47.5),
                        output_path = "inputs")
```
"""
function bathymetry_dataset(; source = nothing,
                            lon_range::Tuple{Real, Real},
                            lat_range::Tuple{Real, Real},
                            tier = nothing,
                            output_path::Union{Nothing, AbstractString} = nothing,
                            force::Bool = false,
                            verbose::Bool = true,
                            kwargs...)
    resolved = preferred_bathymetry_source(; source = source, lon_range = lon_range,
                                           lat_range = lat_range)
    _check_bathymetry_tier(resolved, tier)

    # Where the product goes. A directory holds the source's canonical artefact; a file
    # path is the caller's own and is used as given. The two sources write different
    # kinds of artefact -- a NetCDF file and a Zarr group -- so they are never given the
    # same path: a product written by one source must not be the path the other source
    # writes to, which is how a run ends up failing on a store that is already there.
    destination = if output_path === nothing
        joinpath("inputs", bathymetry_artefact(resolved))
    elseif isdir(output_path) || isempty(splitext(output_path)[2])
        mkpath(output_path)
        joinpath(output_path, bathymetry_artefact(resolved))
    else
        String(output_path)
    end

    # One artefact per (source, range, output_path). The cache is keyed by path, so a run
    # that changed its source must not read the previous product's file back as if it were
    # this one's: `force` is the escape, and a stale source is reported rather than used.
    # The default destination carries no range, so it is never read back: it is written,
    # not trusted.
    if output_path !== nothing && !force && is_cached(destination)
        verbose && println("Using existing bathymetry store: $(destination) " *
                           "(source $(resolved))")
        return geoload(destination)
    end

    # The fetchers all require a real path, because they all download to a file before they
    # can read it. `output_path = nothing` asks for no durable artefact, so it is resolved to
    # a scratch destination here rather than passed down as `nothing` -- which every one of
    # them rejects with a TypeError, so the documented "nothing" path could never have run.
    mkpath(dirname(destination))

    ds = if resolved === :etopo
        # Store first: the cached relief tile is subset to the domain. If the store has no
        # covering tile, download via GeoDataSources' ERDDAP choice (which caches to
        # `destination`). Either way the run's cached sea floor ends up at `destination`.
        d = _etopo_from_store_tile(; lon_range = lon_range, lat_range = lat_range, verbose = verbose)
        if d !== nothing
            # The format follows the name, as it does everywhere else: `geosave` infers the
            # backend from the extension. Forcing NetCDF here wrote a NetCDF *file* to a
            # destination named `.zarr`, which the next reader opens as a directory.
            geosave(destination, d; overwrite = true)
            d
        else
            verbose && println("ETOPO: downloading via ERDDAP (store has no covering tile)...")
            fetch_erddap_bathymetry(; lon_range = lon_range, lat_range = lat_range,
                                    output_path = destination, force = force, verbose = verbose)
        end
    elseif resolved === :nwa
        fetch_nwa_bathymetry(; lon_range = lon_range, lat_range = lat_range,
                             output_path = destination, verbose = verbose,
                             force = force, kwargs...)
    else
        declared = join(sort!(collect(keys(BATHYMETRY_SOURCES)); by = string), ", ")
        error("No fetcher is registered for bathymetry source $(resolved). " *
              "Declared sources: $(declared).")
    end

    if output_path !== nothing && !is_cached(destination)
        error("Bathymetry source $(resolved) was resolved for " *
              "($(lon_range), $(lat_range)) but nothing was written to $(destination). " *
              "The run cannot report a sea floor it does not have.")
    end
    return ds
end

"""
    bathymetry_field(; source::Symbol = :etopo, lon_range, lat_range, tier = nothing,
                     varname::AbstractString = "elevation", kwargs...) -> NamedTuple

Bathymetry for a domain as plain arrays, by named source: `(elevation, lon, lat)`.

Same dispatch over sources as `bathymetry`, and the same rule that a caller names a source,
a range and a tier while which product that resolves to, what format it is stored in, and
whether it is fetched or already cached are decided here. The difference is only the return
shape: this returns the `(n_lon, n_lat)`-ordered arrays a numerical model consumes, rather
than the `GeoDataset` an analysis layer would.

`varname` is a preference rather than a guarantee -- `bathymetry_arrays` probes it first,
then the accepted spellings, and errors listing what the dataset declares when it has none
of them. No variable name is invented.

`output_path`, when given, makes the resolution durable and shared through
`bathymetry_dataset`, so every reader of a run sees the same sea floor.

# Example
```julia
bathy = bathymetry_field(; source = "scotian_shelf", lon_range = (-68.0, -57.0),
                         lat_range = (42.0, 47.5),
                         output_path = "inputs/bathymetry_active.nc")
bathy.elevation isa Matrix{Float64}
```
"""
function bathymetry_field(; source = nothing,
                          lon_range::Tuple{Real, Real},
                          lat_range::Tuple{Real, Real},
                          tier = nothing,
                          varname::AbstractString = "elevation",
                          output_path::Union{Nothing, AbstractString} = nothing,
                          force::Bool = false,
                          verbose::Bool = true,
                          kwargs...)
    ds = bathymetry_dataset(; source = source, lon_range = lon_range,
                            lat_range = lat_range, tier = tier,
                            output_path = output_path, force = force,
                            verbose = verbose, kwargs...)
    return bathymetry_arrays(ds; varname = varname)
end

"""
    bathymetry_arrays(ds::GeoDataset; varname::AbstractString = "elevation") -> NamedTuple

Read `(elevation, lon, lat)` from an already-loaded bathymetry `GeoDataset`, guaranteeing
`(n_lon, n_lat)` ordering and `Float64` throughout.

This is the format-level part of reading bathymetry, and the reason it lives here rather
than in each caller: it is the one place that knows how a provider may spell the elevation
variable and how the field may be transposed.

Variable selection never invents a name. `varname` is used when the dataset declares it, and
otherwise the accepted spellings are probed in order and the first one the dataset actually
declares wins. When the dataset declares none of them this errors and lists what it does
declare, because a misspelled variable and an absent one are different facts and both are
reported rather than guessed at.

Orientation is resolved from the dimensions the source declared, through
`Tethys.dim_permutation`, not from `size`: on a square grid both spellings of a pair have
the same shape, so transposing from shape is how a silently mis-read stays invisible.

A 1-D field is reshaped to a single column rather than refused, since a one-station
transect is a legitimate bathymetry record.
"""
function bathymetry_arrays(ds::GeoDataset; varname::AbstractString = "elevation")
    probes = unique([varname, "elevation", "altitude", "z", "topo"])
    key = findfirst(p -> p in keys(ds.variables), probes)
    key === nothing && error(
        "No bathymetry variable in this dataset. Tried $(probes); it declares " *
        "$(sort!(collect(keys(ds.variables)))).")
    selected = ds.variables[probes[key]]
    permutation = Tethys.dim_permutation(selected.dims, :lon, :lat)
    lon, _ = Tethys.axis(ds, :lon)
    lat, _ = Tethys.axis(ds, :lat)
    raw = Array{Float64}(selected.data)
    elevation = ndims(raw) == 1 ? reshape(raw, :, 1) : permutedims(raw, permutation)
    length(lon) == size(elevation, 1) && length(lat) == size(elevation, 2) || error(
        "Variable $(repr(probes[key])) is $(size(elevation)) but the :lon axis has " *
        "$(length(lon)) values and the :lat axis has $(length(lat)). A field and the " *
        "axes it is gridded on must agree.")
    return (elevation = elevation, lon = lon, lat = lat)
end

"""
    BATHYMETRY_SOURCES

The bathymetry sources `bathymetry` can dispatch to, as `source => human description`. This
is what `available_data` reports and what an unknown-source error quotes, so adding a source
here is the only step needed to make it requestable.
"""
const BATHYMETRY_SOURCES = Dict{Symbol, String}(
    :etopo => "ETOPO 2022, modelled global relief (failsafe, exists everywhere)",
    :nwa   => "measured Northwest Atlantic shelf soundings (higher resolution, different vertical datum)",
)