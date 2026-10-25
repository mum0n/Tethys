"""
High-level API: querying values at points.
"""

export geovalues, sample_dataset, geovalue, geoprofile, geotimeseries

"""
    sample_dataset(ds::GeoDataset, varnames::Vector{String}; kwargs...) -> Dict{String, Any}

Query values at specific coordinate points.

# Arguments
- `ds`: Source dataset
- `varnames`: Variables to query
- `kwargs`: Coordinate points as `lon = -65.0, lat = 45.0, depth = 50.0, time = ...`

# Returns
Dict mapping variable names to queried values (scalars or arrays).

# Examples
```julia
vals = sample_dataset(ds, ["temperature", "salinity"], lon=-65.0, lat=45.0, depth=50.0)
vals = sample_dataset(ds, ["temperature"], lon=[-65.0, -66.0], lat=[45.0, 46.0])
```
"""
function sample_dataset(ds::GeoDataset, varnames::Vector{String}; kwargs...)
    return values_at(ds, varnames; kwargs...)
end

const geovalues = sample_dataset

"""
    geovalue(ds::GeoDataset, varname::String; kwargs...) -> Any

Query a single value for a single variable.
"""
function geovalue(ds::GeoDataset, varname::String; kwargs...)
    vals = geovalues(ds, [varname]; kwargs...)
    return vals[varname]
end

# Convenience: vertical profile
"""
    geoprofile(ds::GeoDataset, varnames::Vector{String}; lon, lat, time=nothing) -> Dict

Get a vertical profile at a location, returning the depths it used alongside the values.

`time` is passed only when the dataset carries a `:time` dimension, and defaults to the first
value on that axis. Passing it explicitly for a dataset that has no `:time` is an error
naming the dimensions present, rather than a request silently dropped: the old default
(`time = 0.0`) forced an axis many datasets do not have, so every profile of a
time-bathymetry-like set failed.
"""
function geoprofile(ds::GeoDataset, varnames::Vector{String}; lon, lat, time=nothing)
    depths = ds.coords[:depth].data
    kwargs = Dict{Symbol, Any}(:lon => lon, :lat => lat, :depth => depths)
    if haskey(ds.dims, :time)
        kwargs[:time] = something(time, first(ds.coords[:time].data))
    elseif time !== nothing
        error("geoprofile was asked for time = $(time), but the dataset has no :time " *
              "dimension. Dimensions present: $(sort(collect(keys(ds.dims)))).")
    end
    vals = geovalues(ds, varnames; kwargs...)
    return merge(vals, Dict(:depth => depths))
end

# Convenience: time series
"""
    geotimeseries(ds::GeoDataset, varnames::Vector{String}; lon, lat, depth=nothing) -> Dict

Get a time series at a location, over the dataset's `:time` axis.

`depth` is passed only when the dataset carries a `:depth` dimension, and defaults to the first
value on it. Passing it explicitly for a dataset that has no `:depth` is an error, for the
reason `geoprofile` documents.
"""
function geotimeseries(ds::GeoDataset, varnames::Vector{String}; lon, lat, depth=nothing)
    times = ds.coords[:time].data
    kwargs = Dict{Symbol, Any}(:lon => lon, :lat => lat, :time => times)
    if haskey(ds.dims, :depth)
        kwargs[:depth] = something(depth, first(ds.coords[:depth].data))
    elseif depth !== nothing
        error("geotimeseries was asked for depth = $(depth), but the dataset has no :depth " *
              "dimension. Dimensions present: $(sort(collect(keys(ds.dims)))).")
    end
    vals = geovalues(ds, varnames; kwargs...)
    return merge(vals, Dict(:time => times))
end