# ==============================================================================
# biodata.jl -- Benthic sources: bottom temperature, substrate, NWA bathymetry
# ==============================================================================

using GeoInterface
using GeoParquet
using SHA: sha256
using DataFrames, Statistics

# Every raw artefact, in the central Tethys lakehouse — `store_root()/raw`, i.e.
# `projects/lakehouse/Tethys/raw` — so the medallion tiers live in the same shared registry
# as the fetched products (see storage.jl). The package `data/` directory was the temporary
# home before the workflow was complete.
const RAW_DATA_DIR = joinpath(store_root(), "raw")

"""
    BIODATA_RAW

Tier-1 raw data, converted from the sources as delivered to compressed GeoParquet point
tables. The names say what the data are and where they come from, because a file called
`bt.rdata` tells a reader nothing and `nwa.chs15sec_lon_lat_depth.xyz` tells a reader more
than it needs to know to be readable.

Every row is preserved - no filtering, no binning, no unit changes - so tier 1 is still the
original archive, just in a container that can be read without streaming 700 MB of text to
get at 30 rows.
"""
const BIODATA_RAW = Dict{Symbol, String}(
    :bottom_temperature => "scotian_shelf_bottom_temperature_surveys_1970_2025.parquet",
    :substrate          => "scotian_shelf_substrate_grain_size.parquet",
    :nwa_bathymetry     => "northwest_atlantic_chs_15arcsec_soundings.parquet",
)

"""
    raw_path(key::Symbol) -> String

Absolute path to the raw artefact for `key`, erroring when it is absent. Absence is a
defect in the package, not something to work around.
"""
function raw_path(key::Symbol)
    name = get(BIODATA_RAW, key) do
        error("Unknown raw source $(key). Available: $(sort(string.(collect(keys(BIODATA_RAW))))).")
    end
    path = joinpath(RAW_DATA_DIR, name)
    isfile(path) || error("Raw file missing from this package: $(path). " *
                          "It is tier-1 data and is expected to be shipped with GeoDataSources.")
    return path
end

"""
    raw_data_digests() -> Dict{Symbol, String}

SHA-256 of every raw artefact. This is the record that answers "which version of the input
produced this cube" without shipping 700 MB through the provenance file.
"""
function raw_data_digests()::Dict{Symbol, String}
    return Dict{Symbol, String}(key => bytes2hex(open(sha256, raw_path(key)))
                                for key in sort!(collect(keys(BIODATA_RAW)); by = string))
end

"""
    bin_points(points, target_lon, target_lat; value=:t, times=nothing,
               time_bins=nothing) -> (field, counts)

Bin irregularly distributed point observations onto a regular grid.

Observations are assigned to the nearest bin centre (`Float64` comparisons, no search on
sorted axes needed), and a bin's value is the mean of its observations. `counts` is returned
with the field because an empty bin (0 observations, `missing` value) and a bin with one
observation both mean "sparse" and must be distinguishable from "zero degrees".

Time is a first-class axis here, not an afterthought: bottom observations span 1970-2025
and collapsing them silently into a climatology is the difference between a field and an
average.

`time_bins` are the **bin labels**: an observation lands in the bin whose label is the
largest one not exceeding it, and anything before the first label is clamped into the first
bin. Passing the years themselves as labels is therefore the common case - one bin per year
- which is what `fetch_bottom_temperature` does, and a mid-year observation joins that year.
"""
function bin_points(
    points::AbstractVector{<:NamedTuple},
    target_lon::AbstractVector{<:Real},
    target_lat::AbstractVector{<:Real};
    value::Symbol = :t,
    times::Union{Nothing, Symbol} = nothing,
    time_bins::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    n_lon, n_lat = length(target_lon), length(target_lat)
    issorted(target_lon) && issorted(target_lat) || error(
        "bin_points needs ascending target axes; reverse a descending latitude axis by " *
        "hand rather than having it guessed for you.")
    lon0, lat0 = Float64(target_lon[1]), Float64(target_lat[1])
    dlon = Float64(n_lon > 1 ? target_lon[2] - target_lon[1] : 1.0)
    dlat = Float64(n_lat > 1 ? target_lat[2] - target_lat[1] : 1.0)

    """Bin index of one coordinate, or `nothing` when it cannot be placed."""
    function bin_index(v, origin, step, n)
        val = Float64(v)
        # A coordinate that is not a number cannot be placed in a bin. Returning nothing
        # skips it, because clamping a NaN into bin 1 would fabricate a sounding there.
        isfinite(val) || return nothing
        idx = round(Int, (val - origin) / step) + 1
        return 1 <= idx <= n ? idx : nothing
    end

    if times === nothing
        sum = zeros(Float64, n_lon, n_lat)
        cnt = zeros(Int, n_lon, n_lat)
        for p in points
            v = getproperty(p, value)
            (ismissing(v) || !isfinite(Float64(v))) && continue
            i = bin_index(getproperty(p, :lon), lon0, dlon, n_lon)
            i === nothing && continue
            j = bin_index(getproperty(p, :lat), lat0, dlat, n_lat)
            j === nothing && continue
            sum[i, j] += Float64(v)
            cnt[i, j] += 1
        end
        return map((s, c) -> c == 0 ? missing : s / c, sum, cnt), cnt
    end

    n_t = length(time_bins)
    sum = zeros(Float64, n_lon, n_lat, n_t)
    cnt = zeros(Int, n_lon, n_lat, n_t)
    for p in points
        v = getproperty(p, value)
        (ismissing(v) || !isfinite(Float64(v))) && continue
        tv = getproperty(p, times)
        (ismissing(tv) || !isfinite(Float64(tv))) && continue
        i = bin_index(getproperty(p, :lon), lon0, dlon, n_lon)
        i === nothing && continue
        j = bin_index(getproperty(p, :lat), lat0, dlat, n_lat)
        raw_k = searchsortedlast(time_bins, Float64(tv))
        k = max(1, min(raw_k, n_t))
        sum[i, j, k] += Float64(v)
        cnt[i, j, k] += 1
    end
    return map((s, c) -> c == 0 ? missing : s / c, sum, cnt), cnt
end

"""
    _benthic_grid(lon_range, lat_range, target_lon, target_lat, resolution_deg) -> (lon, lat)

The grid a fetcher bins onto.

An explicit `target_lon` / `target_lat` wins: the cube owns its grid geometry, because the
cube's own `lon` and `lat` variables are the contract every field in it must satisfy, and a
source gridding itself at a resolution nobody asked for produces a cube whose variables
disagree with its axes.

Failing that, a regular grid is derived from `resolution_deg`, defaulting to 1/12 degree
(GLORYS12V1's resolution). Longitude and latitude are derived separately, because a degree
of longitude is shorter than a degree of latitude by `cos(latitude)`: a grid asked for in
kilometres is not square if it is built in degrees, and only one of those is what a caller
meant.
"""
function _benthic_grid(lon_range, lat_range, target_lon, target_lat, resolution_deg)
    if (target_lon === nothing) != (target_lat === nothing)
        error("Give both target_lon and target_lat, or neither.")
    end
    target_lon !== nothing && return (collect(Float64, target_lon),
                                      collect(Float64, target_lat))

    res = resolution_deg === nothing ? 1.0 / 12.0 : Float64(resolution_deg)
    res > 0 || error("resolution_deg must be positive, got $(res)")
    lon = collect(range(Float64(lon_range[1]), Float64(lon_range[2]), step = res))
    lat = collect(range(Float64(lat_range[1]), Float64(lat_range[2]), step = res))
    return (lon, lat)
end

"""
    _is_point(g) -> Bool

Whether a geometry is a point, whatever its concrete type.

A GeoParquet table read back returns `WellKnownBinary`, not `GeoInterface.Point`, so an
`isa GeoInterface.Point` test always fails on round-tripped data and every coordinate comes
back `NaN`. The coordinates themselves always work on a point, so this is the test that
answers the question actually being asked.
"""
_is_point(g) = try
    GeoInterface.trait(g) == GeoInterface.PointTrait()
catch
    false
end


"""
    raw_points(key::Symbol) -> DataFrame

Read a tier-1 raw point table as a `DataFrame`, with `lon` and `lat` as columns.

The geometry column holds the coordinates; this is the one place that turns them back into
`lon`/`lat` columns, because the ingest and gridding steps that follow index by coordinate
value and should not each re-derive it from a geometry.
"""
function raw_points(key::Symbol)
    df = GeoParquet.read(raw_path(key))
    isempty(df) && error("Tier-1 table for $(key) is empty: $(raw_path(key))")
    # `lon`/`lat` may already be columns: reading is not a one-shot operation, and adding
    # them twice is an argument error rather than a no-op.
    if !(:lon in propertynames(df))
        geom = df[!, :geometry]
        insertcols!(df, :lon => Float64[_is_point(x) ? Float64(GeoInterface.x(x)) : NaN
                                        for x in geom])
        insertcols!(df, :lat => Float64[_is_point(x) ? Float64(GeoInterface.y(x)) : NaN
                                        for x in geom])
    end
    return df
end

"""
    load_bottom_temperature_points() -> DataFrame

Read the raw survey table (tier-1) with the columns Tethys needs.

The survey protocols are heterogeneous, and so is the depth at which each `t` was taken
(`z` spans 0 to about 5000 m, so some rows are mid-water rather than the sea floor). This
function does **not** filter by depth: it returns everything with `z` attached, because
"which rows count as bottom" is a modelling decision the caller owns.
"""
load_bottom_temperature_points() = raw_points(:bottom_temperature)

"""
    load_substrate_points() -> DataFrame

Read the raw substrate table (tier-1).
"""
load_substrate_points() = raw_points(:substrate)

"""
    BenthicField

One binned field destined for a `GeoDataset`: its name, its values, the count of
underlying observations, and the layout it is stored in (`:lon_lat` or `:lon_lat_time`).

Holding the layout on the field rather than inferring it from the array's rank is what lets
one raw file contribute both a static sea-floor depth and a time-varying temperature to the
same dataset without either being reshaped to look like the other.
"""
struct BenthicField
    key::Symbol
    field::AbstractArray
    count::AbstractArray
    layout::Symbol
    units::Any
end

"""
    benthic_dataset(source_key, fields<:AbstractVector{BenthicField}, target_lon,
                    target_lat; time, interpolated, notes) -> GeoDataset

Build a `GeoDataset` from one raw source's binned fields, carrying each field's observation
count, the raw data's digest, and the interpolation flag as attributes, so the result is
self-describing.

A missing bin is `NaN` in the array *and* zero in the count: an unsampled cell must not be
readable as a zero-depth or zero-grain-size cell, and only the pair can say which it is.
"""
function benthic_dataset(
    source_key::Symbol,
    fields::AbstractVector{BenthicField},
    target_lon::AbstractVector,
    target_lat::AbstractVector;
    time::Union{Nothing, AbstractVector} = nothing,
    interpolated::Bool = false,
    notes::String = "",
)
    isempty(fields) && error("benthic_dataset needs at least one field.")
    d_lon = Dimension(name = :lon, size = length(target_lon),
                      coords = collect(Float64, target_lon),
                      units = "degrees_east", dim_type = DIM_SPATIAL)
    d_lat = Dimension(name = :lat, size = length(target_lat),
                      coords = collect(Float64, target_lat),
                      units = "degrees_north", dim_type = DIM_SPATIAL)
    d_t = time === nothing ? nothing :
          Dimension(name = :time, size = length(time),
                    coords = collect(Float64, time),
                    units = "seconds since 1970-01-01T00:00:00Z", dim_type = DIM_TEMPORAL)

    crs = CoordinateSystem(crs = "EPSG:4326")
    """Axis tuple for a field, in the source's own row-major order."""
    axes_of(field::BenthicField) =
        field.layout === :lon_lat ? (d_lon, d_lat) :
        field.layout === :lon_lat_time ? d_t === nothing ? (d_lon, d_lat) : (d_lon, d_lat, d_t) :
        error("Unknown benthic layout $(field.layout) for $(field.key). " *
              "Use :lon_lat or :lon_lat_time.")

    vars = Dict{String, GeoArray}()
    for field in fields
        numeric = map(f -> ismissing(f) ? NaN : Float64(f), field.field)
        dims = axes_of(field)
        size(numeric) == Tuple(d.size for d in dims) || error(
            "Field $(field.key) has shape $(size(numeric)) but layout $(field.layout) " *
            "implies $(Tuple(d.size for d in dims)).")
        vars[string(field.key)] = GeoArray(numeric, dims, crs, Dict{String, Any}())
        vars[string(field.key, "_n")] =
            GeoArray(convert(Array{Int32}, collect(field.count)), dims, crs,
                     Dict{String, Any}("units" => "observations averaged per cell",
                                       "for" => string(field.key)))
    end

    dims_seen = Dict(d_lon.name => d_lon, d_lat.name => d_lat)
    d_t === nothing || (dims_seen[:time] = d_t)

    attrs = Dict{String, Any}(
        "source" => "GeoDataSources tier-1 raw data: $(BIODATA_RAW[source_key])",
        "sha256" => raw_data_digests()[source_key],
        "interpolated" => interpolated,
        "fields" => sort!(collect(string.(keys(fields)))),
        "notes" => notes,
    )
    return GeoDataset(vars, Dict{Symbol, GeoArray}(), dims_seen, crs, attrs, nothing,
                      raw_path(source_key))
end

"""
    INGESTED_DATA_DIR

Where the **ingested** tier lives: raw data read, filtered only by the caller's bounds,
and written back as a table. Nothing is binned, clamped, averaged, or unit-converted here.

Four tiers, and the boundaries between them are what make each one checkable:

| # | Tier | Location | Contents |
|:---|:---|:---|:---|
| 1 | **raw** | `lakehouse/Tethys/raw/` | the source files, byte-for-byte; never opened by a writer |
| 2 | **ingested** | `lakehouse/Tethys/ingested/` | the observations as points, filtered by bounds only |
| 3 | **gridded** | `lakehouse/Tethys/<source>/*.zarr` | one source binned onto a caller's grid, with per-cell observation counts |
| 4 | **cube** | `lakehouse/Tethys/cubes/*.zarr` | several gridded sources assembled into one store, harmonised layouts, provenance |

Tier 3 is per-source: `fetch_bottom_temperature` binning bottom temperature, and nothing
else, is the whole of it. Tier 4 is the assembled product: `assimilate_regional_cube`
resolves each variable's source through `fetch_input` - which writes tier 3 - loads it, and
writes it into the cube. The boundary is a `geoload`, so a cube can be rebuilt from tier 3
without re-binning, and a grid can be changed without re-ingesting.

The ingested tier is the one that was missing: without it, the only place the filtered point
data existed was inside a fetcher's stack frame, so a gridding choice could not be revisited
without re-reading and re-filtering the raw file, and no one could see which observations a
mean came from.
"""
const INGESTED_DATA_DIR = joinpath(store_root(), "ingested")

"""
    ingested_path(key::Symbol, tag::AbstractString = "") -> String

Path for an ingested point table. `tag` distinguishes subsets of the same source.
"""
function ingested_path(key::Symbol, tag::AbstractString = "")
    base = isempty(tag) ? string(key) : "$(key)_$(tag)"
    return joinpath(INGESTED_DATA_DIR, "$(base).parquet")
end

"""
    _write_point_table(path::AbstractString, rows) -> String

Write ingested points as a **GeoParquet point table**, taking either a `DataFrame` or a
vector of `NamedTuple`s.

`lon` and `lat` become the geometry column, which is what the data are - points - so the
ingested tier is a point cloud any GeoParquet reader can open without knowing anything
about GeoDataSources, and every column keeps its own type rather than being flattened to
text and parsed back. Missing values are written as missing, not as `NaN`, so a station
whose depth was never recorded is distinguishable from one whose depth was recorded as
zero.

Any column outside `(:lon, :lat)` is carried through unchanged, so a source gaining a
column needs no change here - and nothing is dropped at this tier.
"""
function _write_point_table(path::AbstractString, rows::AbstractVector)
    isempty(rows) && error("Refusing to write an empty point table at $(path).")
    mkpath(dirname(path))
    cols = [c for c in propertynames(rows[1]) if c != :lon && c != :lat]
    df = DataFrame(:geometry => [GeoInterface.Point(r.lon, r.lat) for r in rows])
    for c in cols
        df[!, c] = [getproperty(r, c) for r in rows]
    end
    GeoParquet.write(path, df, (:geometry,))
    return path
end

function _write_point_table(path::AbstractString, df::DataFrame)
    isempty(df) && error("Refusing to write an empty point table at $(path).")
    mkpath(dirname(path))
    cols = [c for c in names(df) if c ∉ ("lon", "lat")]
    out = DataFrame(:geometry => [GeoInterface.Point(lon, lat)
                                  for (lon, lat) in zip(df.lon, df.lat)])
    for c in cols
        out[!, c] = df[!, c]
    end
    GeoParquet.write(path, out, (:geometry,))
    return path
end

"""
    _read_point_table(path::AbstractString) -> NamedTuple

Read a point table written by `_write_point_table`, returning `(columns, rows)` with the
rows as `NamedTuple`s keyed by column name.

Longitude and latitude are recovered from the geometry column, so a row's fields are
exactly the `columns` the writer was given. Reading through `geoload` on the same path is
also possible and is what the GeoParquet backend does; this reader exists because the
grinding step wants named tuples, not a `GeoDataset`.
"""
function _read_point_table(path::AbstractString)
    isfile(path) || error("Ingested table missing: $(path). Ingest the source first.")
    df = GeoParquet.read(path)
    isempty(df) && error("Ingested table is empty: $(path)")

    others = [Symbol(c) for c in names(df) if c != :geometry]
    isempty(others) && error("Ingested table at $(path) has only geometry and no values.")
    columns = vcat([:lon, :lat], others)

    geom = df[!, :geometry]
    lons = map(geom) do g
        _is_point(g) ? Float64(GeoInterface.x(g)) : Float64(NaN)
    end
    lats = map(geom) do g
        _is_point(g) ? Float64(GeoInterface.y(g)) : Float64(NaN)
    end

    rows = Vector{NamedTuple}(undef, nrow(df))
    for i in 1:nrow(df)
        vals = (lons[i], lats[i], (df[i, c] for c in others)...)
        rows[i] = NamedTuple{Tuple(columns)}(vals)
    end
    return (columns = columns, rows = rows)
end

# ======================================================================
# Tier 2: discretized
# ======================================================================
#
# The chain is raw -> discretized -> modelled -> cube. Tier 2 is where a point
# observation becomes a regular-array value at a *named* resolution, with the
# quality-control decisions that got it there recorded next to the result.
#
# A resolution here is per dataset, not per run: the surveys go on a 1 km grid
# because 1 km is what the station spacing supports, while the CHS soundings
# keep their native 15 arc-second because that is what was measured. A caller
# asking for "bottom temperature" does not get to choose a resolution for
# soundings, and a caller asking for a different one gets an error naming what
# the options are rather than a silently resampled field.

"""
    DISCRETIZATION

Per-dataset discretisation policy: the resolution, its unit, and the reasoning.

The resolution is a property of the data rather than of the caller, because a
grid finer than the station spacing does not add information and a grid coarser
than the measurement spacing throws it away. Putting the number here, where the
data live, is what stops it from being a default argument nobody reads.
"""
const DISCRETIZATION = Dict{Symbol, NamedTuple}(
    :bottom_temperature => (
        resolution = 1.0,
        unit = :km,
        native = "irregular stations, ~1-10 km spacing across 30 programmes",
        rationale = "1 km resolves the station spacing without inventing structure " *
                    "between stations; the counts array records where that matters.",
    ),
    :substrate => (
        resolution = 1.0,
        unit = :km,
        native = "irregular points from an interpolated surface",
        rationale = "matches the grain-size surface's own resolution; finer would " *
                    "reproduce interpolation artefacts, coarser would blur the " *
                    "mud-to-sand transition.",
    ),
    :nwa_bathymetry => (
        resolution = 15.0 / 3600.0,
        unit = :deg,
        native = "15 arc-second soundings",
        rationale = "native resolution: the soundings are the data, and resampling " *
                    "them loses exactly what makes this product better than ETOPO.",
    ),
)

"""
    DISCRETIZED_DATA_DIR

Where tier-2 outputs go: one GeoParquet per (dataset, variable, method), plus a
JSON sidecar recording what was done.
"""
const DISCRETIZED_DATA_DIR = joinpath(store_root(), "discretized")

"""
    write_grid_table(path::AbstractString, lon, lat, field, counts) -> String

Write a tier-2 regular grid as a GeoParquet table: one row per cell, with the cell's
centre as a Point geometry, its value, and how many observations produced it.

The count column is written next to the value rather than in a separate file,
because a cell whose value is `NaN` and a cell that measured zero are different
claims and only the pair can make the distinction.
"""
function write_grid_table(path::AbstractString, lon::AbstractVector, lat::AbstractVector,
                          field::AbstractArray, counts::AbstractArray)
    isempty(field) && error("Refusing to write an empty grid at $(path).")
    mkpath(dirname(path))

    n_lon, n_lat = length(lon), length(lat)
    size(field) == (n_lon, n_lat) && size(counts) == (n_lon, n_lat) ||
        error("Grid fields are $(size(field)) and $(size(counts)), expected " *
              "($(n_lon), $(n_lat)).")

    geoms = GeoInterface.Point[]
    vals = Float64[]
    cnts = Int[]
    for j in 1:n_lat, i in 1:n_lon
        push!(geoms, GeoInterface.Point(lon[i], lat[j]))
        push!(vals, field[i, j])
        push!(cnts, counts[i, j])
    end
    df = DataFrame(:geometry => geoms, :value => vals, :n => cnts)
    GeoParquet.write(path, df, (:geometry,))
    return path
end

"""
    discretized_path(key::Symbol, tag::AbstractString = "") -> String

Path for a tier-2 regular-grid table.
"""
function discretized_path(key::Symbol, tag::AbstractString = "")
    base = isempty(tag) ? string(key) : "$(key)_$(tag)"
    return joinpath(DISCRETIZED_DATA_DIR, "$(base).parquet")
end

"""
    QualityReport

What quality control did to the observations, and therefore what the tier-2
field is worth.

Every field that changed the row count is recorded, because "442 177 observations
were read" is not the same claim as "426 573 were retained" and only the second
one is checkable downstream.
"""
struct QualityReport
    n_read::Int
    n_kept::Int
    dropped::Dict{String, Int}
    notes::String
end

"""
    DiscretizationProvenance

Everything needed to reproduce a tier-2 field without re-deriving it: which raw
file it came from, which QC it passed, which grid it landed on, and which method
produced it. Written as a JSON sidecar next to the field.
"""
struct DiscretizationProvenance
    dataset::Symbol
    variable::String
    method::String
    raw_file::String
    raw_sha256::String
    n_read::Int
    n_kept::Int
    dropped::Dict{String, Int}
    grid_resolution::Float64
    grid_unit::Symbol
    grid_n_lon::Int
    grid_n_lat::Int
    grid_extent::Tuple{Float64, Float64, Float64, Float64}
    value_range::Tuple{Float64, Float64}
    mean_count_per_filled_cell::Float64
    created::String
end

"""
    _qc_bottom_temperature(df::DataFrame) -> (DataFrame, QualityReport)

Quality-control the survey table: drop rows that cannot be used, keep the rest.

A row is dropped only when the value itself is unusable - missing temperature, or
a temperature outside the physically possible range for the ocean. Rows are *not*
dropped for a missing depth (15 604 of them): a bottom observation is a bottom
observation whether or not its depth was recorded, and discarding 3.5% of the
record over a bookkeeping gap is how a dataset quietly loses a decade.

`z` is *not* used to define "bottom" here - that filter is a parameter of
`ingest_bottom_temperature`, because which depth counts as seabed is a modelling
decision the caller owns.
"""
function _qc_bottom_temperature(df::DataFrame)
    n_read = nrow(df)
    dropped = Dict{String, Int}()

    keep_t = .!ismissing.(df.t) .& isfinite.(coalesce.(df.t, NaN))
    dropped["missing_or_nonfinite_temperature"] = n_read - count(keep_t)

    # Freezing point of seawater at depth is about -2.6 C; the surface has been
    # measured at 35 C in shallow lagoons. 40 C is above anything marine.
    t_vals = coalesce.(df.t, NaN)
    plausible = (-5.0 .<= t_vals .<= 40.0) .& keep_t
    dropped["temperature_outside_physical_range"] = count(keep_t) - count(plausible)

    lons = coalesce.(df.lon, NaN); lats = coalesce.(df.lat, NaN)
    on_domain = isfinite.(lons) .& isfinite.(lats) .&
                (-180.0 .<= lons .<= 180.0) .& (-90.0 .<= lats .<= 90.0)
    dropped["coordinates_not_finite_or_in_range"] = count(plausible) - count(plausible .& on_domain)

    ok = plausible .& on_domain
    sub = df[ok, [:lon, :lat, :z, :yr, :t]]
    report = QualityReport(
        n_read, nrow(sub), dropped,
        "QC: dropped missing/non-finite temperature, temperature outside -5..40 C, and " *
        "non-finite or out-of-range coordinates. Missing depth NOT dropped (a bottom " *
        "observation is a bottom observation whether or not its depth was recorded).",
    )
    return sub, report
end

"""
                tag = "", verbose = true) -> DiscretizationProvenance

Discretize one dataset onto its own regular grid, with QC and recorded provenance.

This is the whole of tier 2. It reads tier 1 (`raw_points`), quality-controls it,
bins it onto the resolution the dataset's policy declares, and writes a
GeoParquet field with a JSON sidecar recording every parameter used.

`resolution`, when given, must be equal to the policy resolution - the caller
validates a known resolution rather than silently getting a different one.
"""
function discretize(key::Symbol; lon_range::Tuple{Real, Real},
                    lat_range::Tuple{Real, Real},
                    resolution::Union{Nothing, Real} = nothing,
                    tag::AbstractString = "",
                    verbose::Bool = true,
                    method::String = "bin_mean",
                    variable::String = _default_variable(key))
    policy = get(DISCRETIZATION, key) do
        error("No discretization policy for $(key); add one to DISCRETIZATION.")
    end
    res = resolution === nothing ? policy.resolution : Float64(resolution)
    if resolution !== nothing && res != policy.resolution
        error("Dataset $(key) is discretized at $(policy.resolution) $(policy.unit) by " *
              "policy: $(policy.rationale) Pass resolution=nothing to accept the policy " *
              "value, or override DISCRETIZATION if the policy is wrong.")
    end

    df = raw_points(key)
    if key === :bottom_temperature
        df, report = _qc_bottom_temperature(df)
    else
        # No dataset-specific QC: the bounds are the only filter.
        report = QualityReport(nrow(df), nrow(df), Dict{String,Int}(), "no QC policy defined")
        keep = (lon_range[1] .<= df.lon .<= lon_range[2]) .&
               (lat_range[1] .<= df.lat .<= lat_range[2])
        df = df[keep, [:lon, :lat, variable]]
    end
    verbose && println("  $(key): $(report.n_read) -> $(report.n_kept) observations kept")

    if key === :bottom_temperature
        keep = (lon_range[1] .<= df.lon .<= lon_range[2]) .&
               (lat_range[1] .<= df.lat .<= lat_range[2])
        df = df[keep, [:lon, :lat, :z, :yr, :t]]
        isempty(df) && error("No QC-passing observations in the requested domain.")
    end

    # Per-dataset grid: the resolution, converted to degrees per axis.
    dlat = policy.unit === :km ? res / 110.574 : res
    dlon = policy.unit === :km ? res / (111.320 * cos(deg2rad(0.5 * (lat_range[1] + lat_range[2])))) :
            res * cos(deg2rad(0.5 * (lat_range[1] + lat_range[2])))
    lon = collect(range(Float64(lon_range[1]), step = dlon,
                        length = floor(Int, (lon_range[2] - lon_range[1]) / dlon) + 1))
    lat = collect(range(Float64(lat_range[1]), step = dlat,
                        length = floor(Int, (lat_range[2] - lat_range[1]) / dlat) + 1))

    # Rows as named tuples for `bin_points`, from the QC-passing DataFrame.
    others = key === :bottom_temperature ? [:z, :yr, :t] : [Symbol(variable)]
    rows = Vector{NamedTuple}(undef, nrow(df))
    for (i, r) in enumerate(eachrow(df))
        rows[i] = (lon = Float64(r.lon), lat = Float64(r.lat),
                   (c => Float64(coalesce(r[c], NaN)) for c in others)...)
    end

    # Bin onto the dataset's own grid. Bottom temperature is 3-D (lon, lat, year)
    # because the sea floor changes year to year; a static field is 2-D.
    years = Int[]
    if key === :bottom_temperature
        years = sort!(Int.(unique([Int(r.yr) for r in rows])))
        valfield, counts = bin_points(rows, lon, lat, value = :t, times = :yr,
                                      time_bins = Float64.(years))
    else
        valfield, counts = bin_points(rows, lon, lat, value = others[1])
    end

    n_lon, n_lat = length(lon), length(lat)
    val = map(f -> ismissing(f) ? NaN : Float64(f), valfield)
    finite = filter(isfinite, val)

    # A GeoParquet field: one row per cell (per cell-year for the 3-D case), with the
    # cell centre as the geometry, its value, and how many observations produced it.
    out = discretized_path(key, tag)
    mkpath(dirname(out))
    if ndims(valfield) == 2
        write_grid_table(out, lon, lat, val, counts)
    else
        geoms = GeoInterface.Point[]; vals = Float64[]; cnts = Int[]; yrs = Int[]
        for k in axes(valfield, 3), j in axes(valfield, 2), i in axes(valfield, 1)
            push!(geoms, GeoInterface.Point(lon[i], lat[j]))
            push!(vals, val[i, j, k]); push!(cnts, counts[i, j, k])
            push!(yrs, years[k])
        end
        GeoParquet.write(out, DataFrame(:geometry => geoms, :year => yrs,
                                        :value => vals, :n => cnts), (:geometry,))
    end

    sidecar = replace(out, ".parquet" => ".provenance.json")
    prov = DiscretizationProvenance(
        key, variable, method, basename(raw_path(key)), raw_data_digests()[key],
        report.n_read, report.n_kept, report.dropped, res, policy.unit, n_lon, n_lat,
        (lat_range..., lon_range...),
        isempty(finite) ? (NaN, NaN) : (minimum(finite), maximum(finite)),
        isempty(counts) ? NaN : sum(counts) / count(>(0), counts),
        string(now()),
    )
    open(sidecar, "w") do io
        JSON3.write(io, json_provenance_dict(prov))
    end
    verbose && println("  $(key): $(report.n_read) -> $(report.n_kept) observations; " *
                       "wrote $(out) ($(n_lon)x$(n_lat)) and $(basename(sidecar))")
    return prov
end

"""
    json_provenance_dict(prov::DiscretizationProvenance) -> Dict{String, Any}

A tier-2 provenance record as JSON-ready scalars, so the sidecar has no Julia types.
"""
function json_provenance_dict(prov::DiscretizationProvenance)
    return Dict{String, Any}(
        "tier" => "discretized",
        "dataset" => string(prov.dataset),
        "variable" => prov.variable,
        "method" => prov.method,
        "raw_file" => prov.raw_file,
        "raw_sha256" => prov.raw_sha256,
        "n_read" => prov.n_read,
        "n_kept" => prov.n_kept,
        "dropped" => prov.dropped,
        "quality_report" => prov,
        "grid" => Dict{String, Any}(
            "resolution" => prov.grid_resolution,
            "unit" => prov.grid_unit,
            "n_lon" => prov.grid_n_lon,
            "n_lat" => prov.grid_n_lat,
            "extent" => prov.grid_extent,
        ),
        "value_range" => prov.value_range,
        "mean_count_per_filled_cell" => prov.mean_count_per_filled_cell,
        "created" => prov.created,
    )
end

"""
    _default_variable(key::Symbol) -> String

The value column name for a dataset whose rows carry a single measurement.
"""
_default_variable(key::Symbol) = key === :bottom_temperature ? "t" : "grainsize"

"""
    ingest_bottom_temperature(; lon_range, lat_range, start_year, end_year,
                              max_depth_m, tag) -> String

Read the raw survey table and write the observations that fall inside the requested
`dyear`, `t`, `project` — so nothing is discarded here that a later analysis might want;
filtering is by bounds, and the point of this tier is that it is the *only* transformation.

Returns the path of the written table.
"""
function ingest_bottom_temperature(;
    lon_range::Tuple{Real, Real},
    lat_range::Tuple{Real, Real},
    start_year::Integer = 1970,
    end_year::Integer = 2025,
    max_depth_m::Union{Nothing, Real} = nothing,
    tag::AbstractString = "",
)
    df = raw_points(:bottom_temperature)
    keep = (lon_range[1] .<= df.lon .<= lon_range[2]) .&
           (lat_range[1] .<= df.lat .<= lat_range[2]) .&
           (start_year .<= df.yr .<= end_year)
    if max_depth_m !== nothing
        keep .&= (df.z .<= Float64(max_depth_m))
    end
    sub = df[keep, names(df)]
    isempty(sub) && error("No bottom-temperature observations fall in the requested domain.")

    # Every column the raw table has, in the order it has them. `z` is written as `missing`
    # where the station never recorded a depth - a depth is not looked up from a bathymetry
    # product here, because that is modelling and belongs to a later tier. `date` and
    # `dyear` travel along too, so an analysis can bin by actual date rather than calendar
    # year, and `project` so it can stratify by survey programme.
    path = ingested_path(:bottom_temperature, tag)
    return _write_point_table(path, sub)
end

"""
    ingest_substrate(; lon_range, lat_range, tag = "") -> String

Read the raw substrate table and write the points inside the requested bounds.
"""
function ingest_substrate(; lon_range::Tuple{Real, Real}, lat_range::Tuple{Real, Real},
                          tag::AbstractString = "")
    df = raw_points(:substrate)
    keep = (lon_range[1] .<= df.lon .<= lon_range[2]) .& (lat_range[1] .<= df.lat .<= lat_range[2])
    sub = df[keep, names(df)]
    isempty(sub) && error("No substrate points fall in the requested domain.")
    return _write_point_table(ingested_path(:substrate, tag), sub)
end

"""
    fetch_bottom_temperature(; lon_range, lat_range, resolution_deg = 1/12,
                              start_year = 1970, end_year = 2025,
                              max_depth_m = nothing, output_path = nothing) -> GeoDataset

Bin the raw survey bottom temperature onto a regular `(lon, lat, year)` grid, together with
the bottom depth `z` at which each observation was taken.

The observations come from the **ingested** tier (`ingest_bottom_temperature`), which is the
unmodified point table; this function is the gridding step and is the only place a mean is
taken. That separation is what lets a different grid be requested later without pretending
the binned values are measurements.

`resolution_deg` defaults to GLORYS12V1's 1/12 degree so the surveyed field and the model
field share a grid and can be blended without regridding either. A year with no observation
anywhere is dropped rather than emitted as an all-`NaN` slice, because an empty time slice is
a claim about the sea that nobody made.

`z` is **bottom depth**: the sea-floor depth at the station, which is what makes these
observations bottom observations in the first place. It is binned onto the same `(lon, lat)`
grid as a static field, because the sea floor does not change year to year, and returned as
`survey_depth` alongside the temperature. Depth is therefore available to the cube without
anyone having to infer it from a bathymetry product, and it is kept deliberately separate
from the NWA soundings: reconciling two soundings of the same shelf is a modelling decision,
not a loader's, and happens where the model is rather than here.

`max_depth_m`, when given, keeps only observations in water shallower than it.
"""
function fetch_bottom_temperature(;
    lon_range::Tuple{Real, Real},
    lat_range::Tuple{Real, Real},
    start_year::Integer = 1970,
    end_year::Integer = 2025,
    max_depth_m::Union{Nothing, Real} = nothing,
    output_path::Union{Nothing, AbstractString} = nothing,
    verbose::Bool = true,
    target_lon::Union{Nothing, AbstractVector} = nothing,
    target_lat::Union{Nothing, AbstractVector} = nothing,
    resolution_deg::Union{Nothing, Real} = nothing,
    force::Bool = false,
    backend::Symbol = :zarr,
)
    lon, lat = _benthic_grid(lon_range, lat_range, target_lon, target_lat, resolution_deg)

    table = ingest_bottom_temperature(lon_range = lon_range, lat_range = lat_range,
                                      start_year = start_year, end_year = end_year,
                                      max_depth_m = max_depth_m)
    points = _read_point_table(table).rows
    years = sort!(Int.(unique([Int(r.yr) for r in points])))
    verbose && println("  gridding $(length(points)) ingested observations onto " *
                       "$(length(lon))x$(length(lat)) over $(length(years)) years")

    temp_field, temp_count = bin_points(points, lon, lat, value = :t, times = :yr,
                                        time_bins = Float64.(years))

    # Bottom depth is a property of the station, not of the year, so it is binned over all
    # observations together rather than per year: a depth measured once is a sounding. A
    # station with no recorded depth contributes no sounding, rather than a zero: `z` is a
    # `Union{Missing, Float64}`, so a missing depth is skipped, never read as 0.
    depth_points = [r for r in points if !ismissing(r.z) && isfinite(r.z)]
    depth_field, depth_count = bin_points(depth_points, lon, lat, value = :z)

    ds = benthic_dataset(
        :bottom_temperature,
        [
            BenthicField(:bottom_temperature, temp_field, temp_count, :lon_lat_time,
                         "degrees Celsius"),
            BenthicField(:survey_depth, depth_field, depth_count, :lon_lat,
                         "m (positive down)"),
        ],
        lon, lat,
        time = Float64.(years),
        notes = "Observations: $(ingested_path(:bottom_temperature)) (tier: ingested, " *
                "filtered by bounds only). `z` is bottom depth at the station, binned over " *
                "all years into a static field. Kept separate from the NWA soundings on " *
                "purpose: reconciling two soundings of the same shelf is a modelling " *
                "decision made elsewhere.",
    )
    output_path === nothing && return ds
    geosave(output_path, ds)
    verbose && println("  saved gridded product to $(output_path)")
    return ds
end

"""
    ingest_nwa_bathymetry(; lon_range, lat_range, tag = "") -> String

Stream the raw NWA soundings and write the ones inside the requested bounds.

The 678 MB file is streamed, so nothing here holds it in memory, and the written table is
tab-separated `lon lat z` for consistency with the source format. This is the ingested tier
for the third source: the soundings themselves, filtered by bounds, before any binning.
"""
function ingest_nwa_bathymetry(; lon_range::Tuple{Real, Real},
                               lat_range::Tuple{Real, Real}, tag::AbstractString = "")
    # Tier 1 is a parquet point cloud now, so the filter is a column expression rather
    # than a text scan of 700 MB. The sounding's `z` is written as it was measured, with
    # no depth looked up from anywhere else.
    df = raw_points(:nwa_bathymetry)
    keep = (lon_range[1] .<= df.lon .<= lon_range[2]) .&
           (lat_range[1] .<= df.lat .<= lat_range[2])
    sub = df[keep, ["lon", "lat", "z"]]
    isempty(sub) && error("None of the NWA soundings fall in the requested domain.")
    return _write_point_table(ingested_path(:nwa_bathymetry, tag), sub)
end

"""
    fetch_substrate_grainsize(; lon_range, lat_range, resolution_deg = 1/12,
                               output_path = nothing) -> GeoDataset

Bin the raw substrate record onto a regular grid.

The observations come from the **ingested** tier (`ingest_substrate`), and this function is
the gridding step. Grain size is stored as the provider interpolated it - the raw samples
behind it are not available - so the returned dataset carries `interpolated = true`. Keeping
that flag on the data is the difference between using a measurement grid and using a
surface, and only the data itself can tell you which one you have.
"""
function fetch_substrate_grainsize(;
    lon_range::Tuple{Real, Real},
    lat_range::Tuple{Real, Real},
    output_path::Union{Nothing, AbstractString} = nothing,
    verbose::Bool = true,
    target_lon::Union{Nothing, AbstractVector} = nothing,
    target_lat::Union{Nothing, AbstractVector} = nothing,
    resolution_deg::Union{Nothing, Real} = nothing,
    force::Bool = false,
    backend::Symbol = :zarr,
)
    lon, lat = _benthic_grid(lon_range, lat_range, target_lon, target_lat, resolution_deg)

    table = ingest_substrate(lon_range = lon_range, lat_range = lat_range)
    points = _read_point_table(table).rows
    verbose && println("  gridding $(length(points)) ingested substrate points onto " *
                       "$(length(lon))x$(length(lat))")

    field, count = bin_points(points, lon, lat, value = :grainsize)
    ds = benthic_dataset(
        :substrate,
        [BenthicField(:substrate, field, count, :lon_lat, "mm (grain diameter)")],
        lon, lat, interpolated = true,
        notes = "Observations: $(table) (tier: ingested, filtered by bounds only). Raw grab " *
                "samples are not available: this is the provider's interpolated surface, " *
                "and says so rather than passing for measurements.",
    )
    output_path === nothing && return ds
    geosave(output_path, ds)
    verbose && println("  saved gridded product to $(output_path)")
    return ds
end
"""
    fetch_nwa_bathymetry(; lon_range, lat_range, resolution_deg = 1/12, elevation = true,
                          output_path = nothing) -> GeoDataset

Bin the 15-arc-second NWA soundings (`nwa.chs15sec_lon_lat_depth.xyz`) onto a regular grid.

The file is 28 million tab-separated `lon lat depth` lines, which is streamed rather than
read: nothing here holds it in memory, because 678 MB of ASCII is not a dataset and
materialising one to read the other is how a simple reader becomes a 20 GB process.

`elevation = true` returns elevation positive-up (the file's own convention) with
`bottom_depth` derived; `false` returns positive-down metres instead.

Two datums, and they are not the same sea floor. ETOPO 2022 is a global 15 arc-second
model; these are CHS soundings of the Northwest Atlantic. They will disagree on the shelf
by metres to tens of metres, and neither is wrong. Choose one per run and record which,
rather than averaging them into a bathymetry that came from nowhere.
"""
function fetch_nwa_bathymetry(;
    lon_range::Tuple{Real, Real},
    lat_range::Tuple{Real, Real},
    resolution_deg::Real = 1.0 / 12.0,
    elevation::Bool = true,
    output_path::Union{Nothing, AbstractString} = nothing,
    verbose::Bool = true,
    force::Bool = false,
    backend::Symbol = :zarr,
)
    step = Float64(resolution_deg)
    lon = collect(range(Float64(lon_range[1]), Float64(lon_range[2]), step = step))
    lat = collect(range(Float64(lat_range[1]), Float64(lat_range[2]), step = step))
    n_lon, n_lat = length(lon), length(lat)
    lon0, lat0 = lon[1], lat[1]

    # Ingest first, then grid. The soundings inside the bounds are written once to the
    # ingested tier; gridding them is a separate step so a different resolution does not
    # mean re-streaming 678 MB.
    table = ingest_nwa_bathymetry(lon_range = lon_range, lat_range = lat_range)
    points = _read_point_table(table).rows
    verbose && println("  gridding $(length(points)) ingested NWA soundings onto " *
                       "$(n_lon)x$(n_lat)")

    # A bin's value is the mean of its soundings. An empty bin is NaN, not zero: zero here
    # would be sea level, which is a real depth and not the same as "nothing sounded".
    depth_field, depth_cnt = bin_points(points, lon, lat, value = :z)
    field = map(f -> ismissing(f) ? NaN : Float64(f), depth_field)
    field = elevation ? field : .-field

    filled = count(x -> x > 0, depth_cnt)
    verbose && println("  $(filled) of $(n_lon * n_lat) cells filled, " *
                       "$(n_lon * n_lat - filled) empty")

    ds = benthic_dataset(
        :nwa_bathymetry,
        # The field is named for the physical variable, `:elevation`, not for the source
        # key: `bathymetry_arrays` probes `elevation`/`altitude`/`z`/`topo`, so a variable
        # named `nwa_bathymetry` was unreadable by every consumer of `bathymetry_field`,
        # including this package's own documented `source = "scotian_shelf"` example. The
        # source key stays for provenance, in `benthic_dataset`'s first argument and attrs.
        [BenthicField(:elevation, field, depth_cnt, :lon_lat,
                      elevation ? "m (elevation, positive up)" : "m (depth, positive down)")],
        lon, lat,
        notes = "Observations: $(table) (tier: ingested, bounds-filtered soundings). " *
                "Different vertical datum from ETOPO; they are different products and are " *
                "not interchangeable.",
    )
    output_path === nothing && return ds
    geosave(output_path, ds)
    verbose && println("  saved gridded product to $(output_path)")
    return ds
end

export BIODATA_RAW, RAW_DATA_DIR, raw_path, raw_data_digests, bin_points, BenthicField,
    benthic_dataset, INGESTED_DATA_DIR, ingested_path, ingest_bottom_temperature,
    ingest_substrate, ingest_nwa_bathymetry,
    load_bottom_temperature_points, load_substrate_points,
    fetch_bottom_temperature, fetch_substrate_grainsize, fetch_nwa_bathymetry
