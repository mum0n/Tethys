"""
High-level API: convenience selectors built on `geoselect`.
"""

export geonearest

# geoselect itself lives in api/slice.jl, next to geoslice: they share one contract and
# differ only in the name callers use.

"""
    geonearest(ds::GeoDataset; lon, lat, depth=nothing, time=nothing) -> GeoDataset

Select the dataset at the point nearest the given coordinates.

Dimensions the dataset does not have are ignored rather than defaulted to zero: `depth`
and `time` are omitted unless passed.
"""
function geonearest(ds::GeoDataset; lon, lat, depth = nothing, time = nothing)
    kwargs = Dict{Symbol, Any}(:lon => lon, :lat => lat)
    depth === nothing || (kwargs[:depth] = depth)
    time === nothing || (kwargs[:time] = time)
    return geoselect(ds; kwargs...)
end
