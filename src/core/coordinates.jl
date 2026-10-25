"""
Generic coordinate system utilities for Tethys.

Cartesian by default. Geographic specialization in GeoDataGeoCoordinates.
"""
module Coordinates

using ..Types
using LinearAlgebra
using Dates

# ============================================================
# Generic dimension handling (no CF assumptions)
# ============================================================

"""
    standardize_dimension_name(name::Symbol; mapping::Dict{Symbol,Symbol}=Dict()) -> Symbol

Map dimension name variants to canonical names using optional mapping.
Defaults to identity (no standardization).
"""
const DEFAULT_DIMENSION_MAPPINGS = Dict{Symbol, Symbol}(
    :lon => :lon, :longitude => :lon, :x => :lon, :nav_lon => :lon,
    :lat => :lat, :latitude => :lat, :y => :lat, :nav_lat => :lat,
    :depth => :depth, :lev => :depth, :level => :depth, :z => :depth,
    :time => :time, :t => :time, :date => :time, :datetime => :time
)

"""
    AXIS_FIELDS

Accepted spellings per canonical axis, ordered most-preferred first: the inverse of
`DEFAULT_DIMENSION_MAPPINGS`, exposed so a caller can list the ways a source may name an
axis rather than rebuilding the list locally. One list in the repo, not one per module.

Note `:elevation` is deliberately *not* an alias for `:depth`: elevation is a measured
sea-floor surface and depth is a coordinate of the water column, and conflating them
silently mixes the sign convention of one with the other.
"""
# Compute `AXIS_FIELDS` from `DEFAULT_DIMENSION_MAPPINGS`. A function rather than a bare
# `let` so the value is computed once and cached in the constant below.
function _axis_fields()
    fields = Dict{Symbol, Vector{Symbol}}()
    for (variant, canonical) in DEFAULT_DIMENSION_MAPPINGS
        push!(get!(fields, canonical, Symbol[]), variant)
    end
    for (_, v) in fields
        sort!(v; by = string)
    end
    return fields
end

const AXIS_FIELDS = _axis_fields()

function standardize_dimension_name(name::Symbol; mapping::Dict{Symbol,Symbol}=Dict{Symbol,Symbol}())
    if !isempty(mapping) && haskey(mapping, name)
        return mapping[name]
    end
    return get(DEFAULT_DIMENSION_MAPPINGS, name, name)
end

"""
    axis(ds::GeoDataset, name::Symbol) -> (Vector{Float64}, Dimension)

The coordinate values of axis `name` and its dimension record.

Resolved through `standardize_dimension_name`, so `:longitude`, `:x` and `:nav_lon` all
find the longitude axis, and whatever is returned is in the dataset's own units: no unit
conversion is applied here. Raises when the dataset has no such axis -- an absent axis is
a fact about the dataset, not something to invent.
"""
function axis(ds::GeoDataset, name::Symbol)
    canonical = standardize_dimension_name(name)
    if haskey(ds.coords, canonical)
        return vec(Float64.(ds.coords[canonical].data)), ds.dims[canonical]
    end
    # A dimension that declares coords IS an axis. Producers are free to carry the
    # coordinate values on the `Dimension` rather than also duplicating them as a
    # coordinate `GeoArray` in `ds.coords`; without this fallback such a dataset (e.g.
    # the benthic and ETOPO bathymetry products) declares a lon it then cannot read.
    if haskey(ds.dims, canonical) &&
       ds.dims[canonical].coords !== nothing &&
       !isempty(ds.dims[canonical].coords)
        return vec(Float64.(ds.dims[canonical].coords)), ds.dims[canonical]
    end
    error("Dataset has no $(canonical) axis. It has: $(sort(collect(keys(ds.coords)))).")
end

"""
    dim_permutation(dims, wanted::Symbol...) -> Vector{Int}

Where each name in `wanted` sits in `dims`, in the order the caller wants it.

Returns the positions of the requested names only, so a caller may take a subset of a
variable's axes - WOA23's `t_an` spans `[:lon, :lat, :depth, :time]` and its interpolator
wants the first three. Errors when a wanted name is absent or appears twice: a caller that
asked for three axes and received a permutation into the wrong one would read the wrong data
rather than fail.

This is how a reader resolves orientation: from the dimensions the source actually
declared, not from `size`. On a square grid both spellings of a pair have the same shape,
so guessing from size is how a transposed read stays invisible.
"""
function dim_permutation(dims, wanted::Symbol...)
    names = [standardize_dimension_name(d.name) for d in dims]
    pos = Int[]
    for w in wanted
        i = findfirst(==(w), names)
        if i === nothing
            error("Dimension $(repr(w)) not found among $(names). Declare the axis, or " *
                  "call with the axes this dataset has.")
        end
        i in pos && error("Dimension $(repr(w)) appears more than once among $(names).")
        push!(pos, i)
    end
    return pos
end

"""
    normalize_depth(depth::AbstractVector) -> (Vector{Float64}, Bool)

Normalize depth to positive-down convention (geographic specialization).
"""
function normalize_depth(depth::AbstractVector)
    dep = Float64.(depth)
    dep_max = maximum(abs, dep)
    if minimum(dep) >= -0.5 * dep_max
        return dep, false
    else
        return -dep, true
    end
end

# ============================================================
# Generic coordinate matching and indexing
# ============================================================

"""
    find_coord_indices(coords::AbstractVector, target::Real; mode::Symbol = :nearest) -> Int

Find index of target value in coordinate array.

# Modes
- `:nearest` - nearest neighbor
- `:exact` - exact match (throws if not found)
- `:floor` - largest index with coord <= target
- `:ceil` - smallest index with coord >= target
"""
function find_coord_indices(coords::AbstractVector, target::Real; mode::Symbol = :nearest)
    c = Float64.(coords)
    target = Float64(target)

    if mode === :exact
        idx = findfirst(isequal(target), c)
        isnothing(idx) && error("Exact coordinate $target not found in $(c[1])..$(c[end])")
        return idx
    elseif mode === :nearest
        return argmin(abs.(c .- target))
    elseif mode === :floor
        idx = findlast(x -> x <= target, c)
        isnothing(idx) && error("No coordinate <= $target in $(c[1])..$(c[end])")
        return idx
    elseif mode === :ceil
        idx = findfirst(x -> x >= target, c)
        isnothing(idx) && error("No coordinate >= $target in $(c[1])..$(c[end])")
        return idx
    else
        error("Unknown mode: $mode")
    end
end

"""
    slice_indices(coords::AbstractVector, range::Tuple{Real, Real}; inclusive::Bool = true) -> UnitRange{Int}

Index range covering the coordinates that fall in `range`.

The block is found by comparing values, not by binary search, so the coordinate vector
may be ascending or descending - a latitude axis stored south-to-north and a depth axis
stored surface-down both work. Unsorted coordinates raise: a silently mirrored slice is
worse than an error.
"""
function slice_indices(coords::AbstractVector, range::Tuple{Real, Real}; inclusive::Bool = true)
    c = Float64.(coords)
    lo, hi = Float64.(range)
    lo <= hi || error("Invalid range: $lo > $hi")
    isempty(c) && error("Cannot slice a range [$lo, $hi] on an empty coordinate vector.")
    (issorted(c) || issorted(c; rev = true)) || error(
        "Coordinates must be monotone to slice a range; got $(first(c))..$(last(c)) " *
        "for the axis holding [$(minimum(c)), $(maximum(c))].")

    contained = inclusive ? (lo .<= c .<= hi) : (lo .< c .< hi)
    hits = findall(contained)
    isempty(hits) && error("Empty slice: no coordinates in [$lo, $hi]")
    return first(hits):last(hits)
end

"""
    slice_indices(coords::AbstractVector, values::AbstractVector) -> Vector{Int}

Get indices for exact coordinate values.
"""
function slice_indices(coords::AbstractVector, values::AbstractVector)
    c = Float64.(coords)
    return [find_coord_indices(c, Float64(v); mode = :nearest) for v in values]
end

# ============================================================
# Grid utilities
# ============================================================

"""
    is_regular_grid(coords::AbstractVector; rtol::Real = 1e-10) -> Bool

Check if coordinates form a regularly spaced grid.
"""
function is_regular_grid(coords::AbstractVector; rtol::Real = 1e-10)
    c = Float64.(coords)
    length(c) < 3 && return true
    d = diff(c)
    return all(isapprox.(d, d[1]; rtol = rtol))
end

"""
    grid_spacing(coords::AbstractVector) -> Float64

Get grid spacing (assumes regular grid).
"""
function grid_spacing(coords::AbstractVector)
    c = Float64.(coords)
    length(c) < 2 && return NaN
    return c[2] - c[1]
end

"""
    bounding_box(ds::GeoDataset) -> NamedTuple

Compute the bounding box of a dataset (all dimensions).
"""
function bounding_box(ds::GeoDataset)
    bb = NamedTuple()
    for (dim, d) in ds.dims
        if haskey(ds.coords, dim)
            c = ds.coords[dim].data
            bb = merge(bb, (dim => (Float64(minimum(c)), Float64(maximum(c))),))
        end
    end
    return bb
end

# ============================================================
# Time handling (generic, supports CF-like units)
# ============================================================

"""
    parse_time_units(units::String) -> (epoch::DateTime, calendar::String)

Parse CF time units string like "days since 1990-01-01".
"""
function parse_time_units(units::String)
    m = match(r"(\w+)\s+since\s+([\d\-T:Z]+)", units)
    isnothing(m) && error("Cannot parse time units: $units")
    unit, epoch_str = m.captures
    epoch = DateTime(epoch_str)
    calendar = "standard"
    return epoch, calendar
end

"""
    datetime_to_time(dt::DateTime, units::String) -> Float64

Convert DateTime to numeric time value.
"""
function datetime_to_time(dt::DateTime, units::String)
    epoch, _ = parse_time_units(units)
    unit = lowercase(first(split(units, ' ')))
    delta = dt - epoch
    return _delta_to_unit(delta, unit)
end

function _delta_to_unit(delta::DateTime, unit::String)
    if unit in ("second", "seconds", "s")
        return Float64(Millisecond(delta)) / 1000
    elseif unit in ("minute", "minutes", "min")
        return Float64(Millisecond(delta)) / 60000
    elseif unit in ("hour", "hours", "h")
        return Float64(Millisecond(delta)) / 3600000
    elseif unit in ("day", "days", "d")
        return Float64(Millisecond(delta)) / 86400000
    else
        error("Unsupported time unit for conversion: $unit")
    end
end

export standardize_dimension_name,
    axis,
    dim_permutation,
    normalize_depth,
    find_coord_indices,
    slice_indices,
    is_regular_grid,
    grid_spacing,
    bounding_box,
    parse_time_units,
    datetime_to_time,
    DEFAULT_DIMENSION_MAPPINGS,
    AXIS_FIELDS,
    variables_like,
    is_cached

"""
    variables_like(ds::GeoDataset, keywords::Vector{String}) -> (String, GeoArray)

The first variable whose name matches one of `keywords` (case-insensitive substring).

Providers name the same field differently - `elevation`, `altitude`, `z`, `topo`,
`bedrock_altitude` - and guessing one name per dataset is why a reader works for one
file and fails for the next. Order matters: the first match wins.
"""
function variables_like(ds::GeoDataset, keywords::Vector{String})
    for keyword in keywords
        needle = lowercase(keyword)
        for (name, _) in ds.variables
            occursin(needle, lowercase(name)) && return (name, ds.variables[name])
        end
    end
    error("No variable matching any of $(keywords) in this dataset. Available: " *
          "$(sort(collect(keys(ds.variables)))).")
end

"""
    is_cached(path::AbstractString) -> Bool

Whether `path` names something that exists, whatever it is.

Use this before `isfile`: a Zarr store, an extracted directory, and a single file are
all valid caches, and `isfile` silently says "missing" for two of them.
"""
is_cached(path::AbstractString) = ispath(path)

end # module Coordinates
