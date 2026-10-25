"""
High-level API: regridding.
"""

export georegrid, regrid_dataset, georegrid_regular

"""
    regrid_dataset(ds::GeoDataset, target_grid::GeoDataset; method::Symbol=:bilinear, vars::Vector{String}=String[]) -> GeoDataset

Regrid dataset to a target grid.

# Arguments
- `ds`: Source dataset
- `target_grid`: Dataset defining target coordinates
- `method`: Interpolation method (`:bilinear`, `:nearest`)
- `vars`: Variables to regrid (empty = all data variables)

# Returns
Regridded `GeoDataset`.

# Examples
```julia
ds_regrid = regrid_dataset(ds, target_grid)
ds_regrid = regrid_dataset(ds, target_grid, method=:nearest, vars=["temperature"])
```
"""
function regrid_dataset(ds::GeoDataset, target_grid::GeoDataset; method::Symbol=:bilinear, vars::Vector{String}=String[])
    return regrid(ds, target_grid; method=method, vars=vars)
end

const georegrid = regrid_dataset

# Convenience: regrid to regular lat/lon grid
"""
    georegrid_regular(ds::GeoDataset; lon::AbstractVector, lat::AbstractVector, depth::AbstractVector=Float64[], method::Symbol=:bilinear) -> GeoDataset

Regrid to a regular lat/lon/depth grid.
"""
function georegrid_regular(ds::GeoDataset; lon::AbstractVector, lat::AbstractVector, depth::AbstractVector=Float64[], method::Symbol=:bilinear)
    # Build target grid dataset
    coords = Dict{Symbol, GeoArray}()
    dims = Dict{Symbol, Dimension}()
    
    lon_dim = Dimension(name=:lon, size=length(lon), coords=lon, units="degrees_east")
    lat_dim = Dimension(name=:lat, size=length(lat), coords=lat, units="degrees_north")
    coords[:lon] = GeoArray(lon, (lon_dim,), CoordinateSystem(), Dict{String, Any}())
    coords[:lat] = GeoArray(lat, (lat_dim,), CoordinateSystem(), Dict{String, Any}())
    dims[:lon] = lon_dim
    dims[:lat] = lat_dim
    
    if !isempty(depth)
        dep_dim = Dimension(name=:depth, size=length(depth), coords=depth, units="meters", standard_name="depth")
        coords[:depth] = GeoArray(depth, (dep_dim,), CoordinateSystem(), Dict{String, Any}())
        dims[:depth] = dep_dim
    end
    
    target = GeoDataset(Dict{String, GeoArray}(), coords, dims, ds.crs, Dict{String, Any}(), ds.backend, "regular_grid")
    
    return georegrid(ds, target; method=method)
end