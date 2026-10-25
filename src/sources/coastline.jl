# ==============================================================================
# coastline.jl -- Natural Earth coastline fetch, load, and polygon operations
# ==============================================================================

using Downloads
using JSON3
using GeoInterface
using GeoParquet
using DataFrames

"""
    fetch_natural_earth_coastline(;
        lon_range::Tuple{Real, Real},
        lat_range::Tuple{Real, Real},
        resolution::AbstractString = "10m",
        output_path::AbstractString = joinpath("inputs", "coastline.parquet"),
        margin_deg::Real = 1.0,
        backend::Symbol = :geoparquet,
        verbose::Bool = true
    ) -> GeoDataset

Fetch Natural Earth coastline as GeoParquet via Tethys.

# Arguments
- `lon_range`: Longitude bounds (min, max)
- `lat_range`: Latitude bounds (min, max)
- `resolution`: Natural Earth resolution ("10m", "50m", "110m")
- `output_path`: Output file path (GeoParquet)
- `margin_deg`: Margin in degrees around bounds
- `backend`: Storage backend (:geoparquet, :zarr)
- `verbose`: Print progress messages

# Returns
GeoDataset with coastline LineString geometries.
"""
function fetch_natural_earth_coastline(;
    lon_range::Tuple{Real, Real},
    lat_range::Tuple{Real, Real},
    resolution::AbstractString = "10m",
    output_path::AbstractString = joinpath("inputs", "coastline.parquet"),
    margin_deg::Real = 1.0,
    backend::Symbol = :geoparquet,
    force::Bool = false,
    verbose::Bool = true
)
    if !force && is_cached(output_path)
        verbose && println("Using cached coastline at $(output_path)")
        return geoload(output_path; backend = backend)
    end

    res = lowercase(strip(resolution))
    res in ("10m", "50m", "110m") || error(
        "Natural Earth resolution \"$(resolution)\" must be 10m, 50m, or 110m")

    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/" *
          "ne_$(res)_coastline.geojson"

    mkpath(dirname(output_path))
    lo_lon, hi_lon = minmax(Float64(lon_range[1]), Float64(lon_range[2]))
    lo_lat, hi_lat = minmax(Float64(lat_range[1]), Float64(lat_range[2]))
    lo_lon -= Float64(margin_deg); hi_lon += Float64(margin_deg)
    lo_lat -= Float64(margin_deg); hi_lat += Float64(margin_deg)

    if verbose
        println("Retrieving Natural Earth $(res) coastline...")
        println("  bbox: lon [$(lo_lon), $(hi_lon)], lat [$(lo_lat), $(hi_lat)]")
    end

    # Download and parse GeoJSON
    text = Downloads.download(url)
    gj = JSON3.read(text)

    # Build DataFrame for GeoParquet
    rows = []
    feature_id = 0
    for feat in gj.features
        geom = feat.geometry
        if geom.type == "MultiLineString"
            for (ring_idx, ring) in enumerate(geom.coordinates)
                feature_id += 1
                lons = Float64[pt[1] for pt in ring]
                lats = Float64[pt[2] for pt in ring]
                # Check if any point is in bbox
                any(lo_lon .<= lons .<= hi_lon) && any(lo_lat .<= lats .<= hi_lat) || continue
                
                # Create LineString geometry
                coords = [(lons[i], lats[i]) for i in 1:length(lons)]
                line = GeoInterface.LineString(coords)
                
                props = hasproperty(feat, :properties) ? feat.properties : nothing
                nm = (isnothing(props) || !haskey(props, :name)) ? "coastline" : String(props[:name])
                code = Symbol(lowercase(replace(nm, r"[^A-Za-z0-9]+" => "_")))
                
                push!(rows, (id=feature_id, name=nm, code=string(code), geometry=line))
            end
        elseif geom.type == "LineString"
            feature_id += 1
            lons = Float64[pt[1] for pt in geom.coordinates]
            lats = Float64[pt[2] for pt in geom.coordinates]
            any(lo_lon .<= lons .<= hi_lon) && any(lo_lat .<= lats .<= hi_lat) || continue
            
            coords = [(lons[i], lats[i]) for i in 1:length(lons)]
            line = GeoInterface.LineString(coords)
            
            props = hasproperty(feat, :properties) ? feat.properties : nothing
            nm = (isnothing(props) || !haskey(props, :name)) ? "coastline" : String(props[:name])
            code = Symbol(lowercase(replace(nm, r"[^A-Za-z0-9]+" => "_")))
            
            push!(rows, (id=feature_id, name=nm, code=string(code), geometry=line))
        end
    end

    if isempty(rows)
        error("No coastline features found in bounding box. Check domain bounds.")
    end

    df = DataFrame(rows)
    GeoParquet.write(output_path, df, (:geometry,))
    
    # Also load as GeoDataset for immediate use
    ds = geoload(output_path; backend=backend)
    
    verbose && println("  kept $(length(rows)) coastline segments -> $(output_path)")
    return ds
end

"""
    load_coastline_geodata(filepath::AbstractString; backend=:geoparquet) -> GeoDataset

Load coastline from GeoParquet file.
"""
function load_coastline_geodata(filepath::AbstractString; backend::Symbol = :geoparquet)
    return geoload(filepath; backend=backend)
end
