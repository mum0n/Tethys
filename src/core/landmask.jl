"""
    landmask.jl

Point-in-polygon evaluation and terrestrial/marine mask determinations based on
geographic boundary geometries.

Functions in this module evaluate spatial points against closed polygon rings.
Network operations and dataset ingestion are separated and reside in `GeoDataSources`.
Boundary geometries are supplied explicitly by callers or loaded from specified files.
"""

import ArchGDAL

export point_in_polygon, is_point_on_land, is_marine_water,
    load_coastline_polygons, DEFAULT_COASTLINE_PATH

"""
    DEFAULT_COASTLINE_PATH

Default relative file path for coastline polygon geometry files (`inputs/coastline.parquet`).
When absent, land masking functions return `false` indicating an undetermined or
unclassified boundary.
"""
const DEFAULT_COASTLINE_PATH = joinpath("inputs", "coastline.parquet")


"""
    load_coastline_polygons(; coastline_path = DEFAULT_COASTLINE_PATH) -> Vector{NamedTuple}

Read coastline boundary polygons from the specified file path. Returns an empty vector
if the file is not accessible.
"""
function load_coastline_polygons(;
    coastline_path::Union{Nothing, AbstractString} = DEFAULT_COASTLINE_PATH
)
    isnothing(coastline_path) && return NamedTuple[]
    return load_coastline_polygons(String(coastline_path))
end

"""
    point_in_polygon(
        x::Real,
        y::Real,
        poly_lons::AbstractVector,
        poly_lats::AbstractVector
    ) -> Bool

Determine whether planar coordinate point \$(x, y)\$ lies within the interior of the
polygon defined by coordinate boundary vectors `(poly_lons, poly_lats)`.

The boundary vertices are evaluated using `ArchGDAL.contains` on an OGR Polygon.
If the initial and terminal vertices do not coincide, the linear ring is closed
automatically prior to evaluation.

# Arguments
- `x::Real`: Point x-coordinate (e.g. longitude in degrees).
- `y::Real`: Point y-coordinate (e.g. latitude in degrees).
- `poly_lons::AbstractVector`: X-coordinates of the polygon boundary vertices.
- `poly_lats::AbstractVector`: Y-coordinates of the polygon boundary vertices.

# Returns
- `Bool`: `true` if the point is strictly inside the polygon interior; `false` otherwise.
"""
function point_in_polygon(
    x::Real,
    y::Real,
    poly_lons::AbstractVector,
    poly_lats::AbstractVector
)::Bool
    n = min(length(poly_lons), length(poly_lats))
    n < 3 && return false

    vx = Float64.(poly_lons[1:n])
    vy = Float64.(poly_lats[1:n])
    if vx[begin] != vx[end] || vy[begin] != vy[end]
        push!(vx, vx[begin])
        push!(vy, vy[begin])
    end

    poly = ArchGDAL.createpolygon(vx, vy)
    pt = ArchGDAL.createpoint(Float64(x), Float64(y))
    return ArchGDAL.contains(poly, pt)
end

"""
    _ring_named_tuple(name, code, coords) -> NamedTuple

One coastline ring as `(name, code, lons, lats)`, from GeoInterface's coordinate
representation. A ring is the first two coordinates of every point: a third is height, not
position along a shoreline.
"""
function _ring_named_tuple(name, code, coords)
    lons = Float64[Float64(pt[1]) for pt in coords]
    lats = Float64[Float64(pt[2]) for pt in coords]
    return (name = name, code = code, lons = lons, lats = lats)
end

"""
    load_coastline_polygons(filepath::AbstractString) -> Vector{NamedTuple}

Load coastline rings from a GeoParquet file. Returns a vector of `NamedTuple`s with
fields `(name, code, lons, lats)` for `LineString` features and individual rings
of `MultiLineString` features. Returns an empty vector if the file does not exist.
"""
function load_coastline_polygons(filepath::AbstractString)
    is_cached(filepath) || return NamedTuple[]
    endswith(filepath, ".parquet") || error(
        "Unsupported coastline file format: $(filepath). Format must be .parquet.")

    ds = load_dataset(filepath; backend = :geoparquet)
    haskey(ds.variables, "geometry") ||
        error("$(filepath) does not contain a 'geometry' variable. Available: " *
              "$(sort(collect(keys(ds.variables)))).")

    geoms = ds.variables["geometry"].data
    names = haskey(ds.variables, "name") ? ds.variables["name"].data :
        fill("", length(geoms))
    codes = haskey(ds.variables, "code") ? ds.variables["code"].data :
        fill("unknown", length(geoms))

    polys = NamedTuple[]
    for i in eachindex(geoms)
        geom = geoms[i]
        name = string(i in eachindex(names) ? names[i] : "")
        code = Symbol(i in eachindex(codes) ? codes[i] : "unknown")

        trait = GeoInterface.trait(geom)
        if trait isa GeoInterface.LineStringTrait
            push!(polys, _ring_named_tuple(name, code, GeoInterface.coordinates(geom)))
        elseif trait isa GeoInterface.MultiLineStringTrait
            for part in GeoInterface.coordinates(geom)
                push!(polys, _ring_named_tuple(name, code, part))
            end
        end
    end
    return polys
end

"""
    is_point_on_land_geodata(
        lon::Real,
        lat::Real;
        coastline::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing,
        coastline_file::Union{Nothing, AbstractString} = nothing
    ) -> Bool

Determine whether geographic coordinate `(lon, lat)` lies on land according to
provided coastline polygon boundaries.

Polygon rings must be geometrically closed (first coordinate equals last coordinate);
open curves evaluate to `false`.

# Arguments
- `lon::Real`: Longitude coordinate in degrees.
- `lat::Real`: Latitude coordinate in degrees.
- `coastline`: Optional collection of polygon named tuples with fields `(:lons, :lats)`.
- `coastline_file`: Optional path to a GeoParquet geometry file.

# Returns
- `Bool`: `true` if the point is contained within any land boundary polygon.
"""
function is_point_on_land(
    lon::Real,
    lat::Real;
    coastline::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing,
    coastlines::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing,
    coastline_file::Union{Nothing, AbstractString} = nothing
)
    polys_source = coastline !== nothing ? coastline : coastlines
    polys = if !isnothing(polys_source)
        polys_source
    elseif !isnothing(coastline_file)
        load_coastline_polygons(coastline_file)
    else
        NamedTuple[]
    end

    isempty(polys) && return false

    # Boundary rings require topological closure for spatial containment tests.
    any(p -> p.lons[begin] != p.lons[end] || p.lats[begin] != p.lats[end], polys) && return false

    x = Float64(lon)
    y = Float64(lat)

    for poly in polys
        min_x, max_x = extrema(poly.lons)
        min_y, max_y = extrema(poly.lats)
        if x >= min_x && x <= max_x && y >= min_y && y <= max_y
            if point_in_polygon(x, y, poly.lons, poly.lats)
                return true
            end
        end
    end

    return false
end

"""
    is_marine_water(
        lon::Real,
        lat::Real;
        bathymetry::Union{Function, GeoDataset, AbstractString, Nothing} = nothing,
        coastlines::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing,
        coastline::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing,
        min_seabed_depth::Real = 0.0
    ) -> Bool

Determine whether geographic coordinate `(lon, lat)` represents marine water.

The point must not intersect terrestrial land polygons. When a bathymetry dataset or
interpolator is supplied, water column depth must also satisfy \$z_{\\text{bed}} \\le -h_{\\min}\$.

# Arguments
- `lon::Real`: Longitude coordinate in degrees.
- `lat::Real`: Latitude coordinate in degrees.
- `bathymetry`: Optional elevation dataset, file path, or `(lon, lat) -> z` function.
- `coastlines` / `coastline`: Optional collection of coastline polygon rings.
- `min_seabed_depth::Real`: Minimum seabed depth threshold in metres (default 0.0 m).

# Returns
- `Bool`: `true` if the point is non-terrestrial and meets depth constraints.
"""
function is_marine_water(
    lon::Real,
    lat::Real;
    bathymetry::Union{Function, GeoDataset, AbstractString, Nothing} = nothing,
    coastlines::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing,
    coastline::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing,
    min_seabed_depth::Real = 0.0,
)
    resolved_coast = coastline !== nothing ? coastline : coastlines
    if is_point_on_land(lon, lat; coastline = resolved_coast)
        return false
    end

    isnothing(bathymetry) && return true

    bathy_fn = bathymetry isa GeoDataset ? get_bathymetry_interpolator(bathymetry) :
               bathymetry isa AbstractString ?
                   get_bathymetry_interpolator(load_dataset(bathymetry)) : bathymetry

    z_bed = bathy_fn(Float64(lon), Float64(lat))
    h_threshold = -max(0.0, Float64(min_seabed_depth))
    return z_bed <= h_threshold
end
