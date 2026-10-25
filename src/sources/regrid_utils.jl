# ==============================================================================
# regrid_utils.jl -- Slicing helpers for fetched datasets
# ==============================================================================

"""
    slice_bathymetry_geodata(ds::GeoDataset; lon_range, lat_range) -> GeoDataset

Slice a bathymetry GeoDataset to a geographic bounding box.
"""
function slice_bathymetry_geodata(ds::GeoDataset;
    lon_range::Tuple{Real, Real} = (-180.0, 180.0),
    lat_range::Tuple{Real, Real} = (-90.0, 90.0))
    return geoslice(ds; lon=lon_range, lat=lat_range)
end

"""
    slice_wind_geodata(ds::GeoDataset; lon_range, lat_range, time_range=nothing) -> GeoDataset

Slice a wind GeoDataset spatially and optionally temporally.
"""
function slice_wind_geodata(ds::GeoDataset;
    lon_range::Tuple{Real, Real} = (-180.0, 180.0),
    lat_range::Tuple{Real, Real} = (-90.0, 90.0),
    time_range::Union{Nothing, Tuple{Real, Real}} = nothing)
    if isnothing(time_range)
        return geoslice(ds; lon=lon_range, lat=lat_range)
    else
        return geoslice(ds; lon=lon_range, lat=lat_range, time=time_range)
    end
end

"""
    extract_grid_coordinates_geodata(ds::GeoDataset) -> NamedTuple

Extract 1D coordinate vectors from a GeoDataset.
"""
function extract_grid_coordinates_geodata(ds::GeoDataset)
    # No axis is optional here: an extractor that silently returns an empty vector for a
    # missing axis hands a caller an empty grid that then fails far away.
    lons = axis(ds, :lon)
    lats = axis(ds, :lat)
    depths = axis(ds, :depth)

    return (lons = lons, lats = lats, depths = depths)
end
