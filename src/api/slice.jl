"""
High-level API: dataset slicing.
"""

export geoslice, slice_dataset, geoselect, select_dataset, geosubset_ids, geosubset_cube, geoslice_bbox

"""
    slice_dataset(data::GeoDataset; kwargs...) -> GeoDataset

Return a subset of `data` selected by coordinate range, coordinate list, or single value.
See `GeoDataOperations.slice` for the keyword contract: a `(min, max)` pair selects a
range, a vector selects those coordinates, and a single value selects a point and drops
the dimension.

`backend` is accepted and ignored: slicing is a generic coordinate operation and runs the
same way for every backend. It is retained so callers do not have to special-case a
dataset they happened to load from Zarr.
"""
function slice_dataset(data::GeoDataset; backend = nothing, kwargs...)
    return slice(data; kwargs...)
end

const geoslice = slice_dataset

"""
    select_dataset(data::GeoDataset; kwargs...) -> GeoDataset

Select specific coordinate values. `select_dataset` and `slice_dataset` share one contract (see
`GeoDataOperations.slice`); both names are kept because callers read differently.
"""
function select_dataset(data::GeoDataset; backend = nothing, kwargs...)
    return select_values(data; kwargs...)
end

const geoselect = select_dataset

"""
    geoslice_bbox(ds::GeoDataset; lon_range=nothing, lat_range=nothing, depth_range=nothing, time_range=nothing, backend=nothing) -> GeoDataset

Slice to a geographic bounding box given in degrees.

A range is applied only when the dataset has that dimension, so a 2-D bathymetry set can
be sliced with the same keyword list as a 4-D temperature set. Ranges outside the
dataset's extent snap inward, never silently expanding it.
"""
function geoslice_bbox(ds::GeoDataset; lon_range = nothing, lat_range = nothing,
                       depth_range = nothing, time_range = nothing, backend = nothing)
    kwargs = Dict{Symbol, Any}()
    for (dim, rng) in ((:lon, lon_range), (:lat, lat_range), (:depth, depth_range),
                       (:time, time_range))
        if rng === nothing
            continue
        end
        if !haskey(ds.dims, dim)
            continue     # the dataset has no such axis; nothing to bound
        end
        kwargs[dim] = rng
    end
    isempty(kwargs) && error(
        "geoslice_bbox got no range that applies to this dataset. Dimensions present: " *
        "$(sort(collect(keys(ds.dims)))).")
    return geoslice(ds; backend = backend, kwargs...)
end

"""
    geosubset_ids(ds::GeoDataset, var::String, coord::Symbol, ids::Vector{Int}) -> GeoDataset

Subset to explicit integer indices along `coord`, in the given order. The indices are
translated to coordinate values, so the result keeps working for every later operation
that selects by coordinate.
"""
function geosubset_ids(ds::GeoDataset, var::String, coord::Symbol, ids::Vector{Int})
    haskey(ds.coords, coord) || error(
        "Cannot subset by :$coord: the dataset has no such coordinate. Coordinates: " *
        "$(sort(collect(keys(ds.coords)))).")
    isempty(ids) && error("geosubset_ids needs at least one index.")
    c = vec(ds.coords[coord].data)
    all(1 .<= ids .<= length(c)) || error("Indices out of range for :$coord (1..$(length(c))).")
    # Selecting by coordinate value keeps the order and any duplicates of `ids`.
    return geoslice(ds; coord => c[ids])
end
