"""
High-level API: spatial/temporal indexing.
"""

export index_dataset, geoindex, geosubset, geosubset_spatial, geosubset_temporal

"""
    index_dataset(ds::GeoDataset; spatial::Bool=true, temporal::Bool=true) -> GeoIndex

Build a spatial/temporal index for fast queries. Runs on the generic index in
`GeoDataOperations.build_index`; backends do not carry their own index.
"""
index_dataset(ds::GeoDataset; spatial::Bool = true, temporal::Bool = true) =
    build_index(ds; spatial = spatial, temporal = temporal)

const geoindex = index_dataset

"""
    geosubset(ds::GeoDataset; bbox::Tuple, time_range::Tuple) -> GeoDataset

Fast spatial/temporal subsetting using index if available.

# Arguments
- `bbox`: Bounding box as `(lon_min, lon_max, lat_min, lat_max)`
- `time_range`: Time range as `(t_min, t_max)`

# Returns
Subsetted `GeoDataset`.
"""
function geosubset(ds::GeoDataset; bbox::Tuple{Real,Real,Real,Real}, time_range::Tuple{Real,Real}=(0, Inf))
    lon_min, lon_max, lat_min, lat_max = bbox
    t_min, t_max = time_range
    return geoslice(ds; lon=(lon_min, lon_max), lat=(lat_min, lat_max), time=(t_min, t_max))
end

# Convenience: spatial subset
"""
    geosubset_spatial(ds::GeoDataset; lon::Tuple, lat::Tuple) -> GeoDataset

Spatial subset by longitude and latitude ranges.
"""
function geosubset_spatial(ds::GeoDataset; lon::Tuple, lat::Tuple)
    return geoslice(ds; lon=lon, lat=lat)
end

# Convenience: temporal subset
"""
    geosubset_temporal(ds::GeoDataset; time::Tuple) -> GeoDataset

Temporal subset by time range.
"""
function geosubset_temporal(ds::GeoDataset; time::Tuple)
    return geoslice(ds; time=time)
end