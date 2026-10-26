"""
    crs.jl

Coordinate reference systems, projections, conformal transformations, vector rotations,
and grid re-projection lattices based on `CoordRefSystems.jl`.
"""
module CRS

using CoordRefSystems
using CoordRefSystems: Shift, WGS84Latest, CRS as CRSSystem, Projected, EPSG, ESRI, string2code
using Unitful: °, m
using Interpolations
using ..Types: GeoDataset, GeoArray, CoordinateSystem
using ..Coordinates: axis, dim_permutation

export
    AbstractProjection,
    GeographicProjection,
    CoordRefSystemsGeographicProjection,
    CoordRefSystemsProjection,
    UTMProjection,
    LambertProjection,
    MercatorProjection,
    RotatedFrame,
    ReprojectionGrid,
    is_planar,
    resolve_crs,
    create_projection,
    forward_project,
    inverse_project,
    grid_convergence_angle,
    geographic_to_rotated,
    rotated_to_geographic,
    rotate_vector,
    unrotate_vector,
    rotate_vector_to_grid,
    rotate_vector_to_geo,
    reproject_to_geographic,
    optimal_shelf_rotation_angle,
    utm_to_lonlat,
    lonlat_to_utm,
    extract_wkt,
    is_wgs84_wkt,
    CoordinateTransformer,
    build_coordinate_transformer,
    transform_point,
    transform_polygon,
    transform_vector_dataset,
    projection_from_crs,
    transform_coordinates

# ==============================================================================
# Geodetic Constants
# ==============================================================================
const EARTH_MEAN_RADIUS = 6371000.0
const EARTH_ROTATION_RATE = 7.292115e-5

# ==============================================================================
# Abstract Type Hierarchy
# ==============================================================================

"""
    AbstractProjection

Abstract base type for coordinate reference systems and map projections.
"""
abstract type AbstractProjection end

"""
    is_planar(proj::AbstractProjection) -> Bool

Determine whether `proj` maps to a planar metric Cartesian coordinate system (meters)
or represents spherical angular coordinates (degrees).
"""
is_planar(::AbstractProjection) = true

# ==============================================================================
# 1. Geographic Coordinate Reference System (Unprojected)
# ==============================================================================

"""
    GeographicProjection <: AbstractProjection

Standard spherical geographic coordinates `(lon, lat)` in degrees East and North.
No metric projection or rotation is applied.
"""
struct GeographicProjection <: AbstractProjection end

is_planar(::GeographicProjection) = false

forward_project(::GeographicProjection, lon::Real, lat::Real) =
    (Float64(lon), Float64(lat))

inverse_project(::GeographicProjection, x::Real, y::Real) =
    (Float64(x), Float64(y))

grid_convergence_angle(::GeographicProjection, lon::Real, lat::Real) = 0.0

"""A geographic CRS wrapper retaining non-WGS84 datum metadata from CoordRefSystems."""
struct CoordRefSystemsGeographicProjection{C} <: AbstractProjection
    crs::C
end

is_planar(::CoordRefSystemsGeographicProjection) = false

function forward_project(proj::CoordRefSystemsGeographicProjection, lon::Real, lat::Real)
    point = convert(proj.crs, LatLon(Float64(lat) * °, Float64(lon) * °))
    return (Float64(point.lon / °), Float64(point.lat / °))
end

function inverse_project(proj::CoordRefSystemsGeographicProjection, x::Real, y::Real)
    point = proj.crs(Float64(x) * °, Float64(y) * °)
    geographic = convert(LatLon, point)
    return (Float64(geographic.lon / °), Float64(geographic.lat / °))
end

grid_convergence_angle(::CoordRefSystemsGeographicProjection, lon::Real, lat::Real) = 0.0

# ==============================================================================
# 2. CoordRefSystems.jl Wrapper Projection
# ==============================================================================

"""
    CoordRefSystemsProjection{C} <: AbstractProjection

Projection wrapper using `CoordRefSystems.jl` for geodetic transformations.

# Fields
- `crs::C`: Underlying `CoordRefSystems.CRS` type.
- `center_x::Float64`: Domain origin false Easting offset in meters.
- `center_y::Float64`: Domain origin false Northing offset in meters.
- `rotation_deg::Float64`: Domain rotation angle relative to grid axes.
- `cos_rot::Float64`: Cosine of `rotation_deg`.
- `sin_rot::Float64`: Sine of `rotation_deg`.
"""
struct CoordRefSystemsProjection{C} <: AbstractProjection
    crs          :: C
    center_x     :: Float64
    center_y     :: Float64
    rotation_deg :: Float64
    cos_rot      :: Float64
    sin_rot      :: Float64
end

function forward_project(proj::CoordRefSystemsProjection, lon::Real, lat::Real)
    pt_ll = LatLon(Float64(lat) * °, Float64(lon) * °)
    pt_proj = convert(proj.crs, pt_ll)
    dx = Float64(pt_proj.x / m) - proj.center_x
    dy = Float64(pt_proj.y / m) - proj.center_y

    if proj.rotation_deg == 0.0
        return (dx, dy)
    end
    x = dx * proj.cos_rot + dy * proj.sin_rot
    y = -dx * proj.sin_rot + dy * proj.cos_rot
    return (x, y)
end

function inverse_project(proj::CoordRefSystemsProjection, x::Real, y::Real)
    dx, dy = if proj.rotation_deg == 0.0
        (Float64(x), Float64(y))
    else
        (
            Float64(x) * proj.cos_rot - Float64(y) * proj.sin_rot,
            Float64(x) * proj.sin_rot + Float64(y) * proj.cos_rot
        )
    end
    raw_x = (dx + proj.center_x) * m
    raw_y = (dy + proj.center_y) * m
    pt_proj = proj.crs(raw_x, raw_y)
    pt_ll = convert(LatLon, pt_proj)
    return (Float64(pt_ll.lon / °), Float64(pt_ll.lat / °))
end

function grid_convergence_angle(proj::CoordRefSystemsProjection, lon::Real, lat::Real)
    dlat = 1e-5
    x0, y0 = forward_project(proj, lon, lat)
    x1, y1 = forward_project(proj, lon, lat + dlat)
    return rad2deg(atan(x1 - x0, y1 - y0))
end

# ==============================================================================
# 3. Rotated Local Frame (Conformal Oblique Tangent Frame)
# ==============================================================================

"""
    RotatedFrame <: AbstractProjection

Local planar conformal coordinate reference frame centered at a specified
geographic origin `(lon_center, lat_center)` with an orientation angle `angle_deg`
measured counter-clockwise from true geographic East.

# Fields
- `lon_center::Float64`: Reference longitude of domain origin in degrees East.
- `lat_center::Float64`: Reference latitude of domain origin in degrees North.
- `angle_deg::Float64`: Rotation angle in degrees, counter-clockwise from East.
- `extent_x::Float64`: Full along-frame spatial dimension in meters.
- `extent_y::Float64`: Full cross-frame spatial dimension in meters.
- `cos_theta::Float64`: Cosine of the rotation angle.
- `sin_theta::Float64`: Sine of the rotation angle.
"""
struct RotatedFrame <: AbstractProjection
    lon_center :: Float64
    lat_center :: Float64
    angle_deg  :: Float64
    extent_x   :: Float64
    extent_y   :: Float64
    cos_theta  :: Float64
    sin_theta  :: Float64

    function RotatedFrame(
        lon_center::Real,
        lat_center::Real,
        angle_deg::Real,
        extent_x::Real = 0.0,
        extent_y::Real = 0.0
    )
        rad = deg2rad(Float64(angle_deg))
        new(
            Float64(lon_center),
            Float64(lat_center),
            Float64(angle_deg),
            Float64(extent_x),
            Float64(extent_y),
            cos(rad),
            sin(rad)
        )
    end
end

RotatedFrame(; lon_center::Real, lat_center::Real, angle_deg::Real = 0.0,
             extent_x::Real = 0.0, extent_y::Real = 0.0) =
    RotatedFrame(lon_center, lat_center, angle_deg, extent_x, extent_y)

"""
    geographic_to_rotated(lon::Real, lat::Real, frame::RotatedFrame) -> (Float64, Float64)

Transform geographic coordinates `(lon, lat)` in degrees East and North into local
Cartesian coordinates `(x, y)` in meters within `frame`.
"""
function geographic_to_rotated(lon::Real, lat::Real, frame::RotatedFrame)
    r_earth = EARTH_MEAN_RADIUS
    dlat_rad = deg2rad(Float64(lat) - frame.lat_center)
    mid_lat_rad = deg2rad((Float64(lat) + frame.lat_center) / 2.0)
    dlon_rad = deg2rad(Float64(lon) - frame.lon_center)

    eta = r_earth * dlat_rad
    xi = r_earth * dlon_rad * cos(mid_lat_rad)

    x = xi * frame.cos_theta + eta * frame.sin_theta
    y = -xi * frame.sin_theta + eta * frame.cos_theta
    return (x, y)
end

"""
    rotated_to_geographic(x::Real, y::Real, frame::RotatedFrame) -> (Float64, Float64)

Transform local rotated Cartesian coordinates `(x, y)` in meters into geographic
coordinates `(lon, lat)` in degrees East and North.
"""
function rotated_to_geographic(x::Real, y::Real, frame::RotatedFrame)
    r_earth = EARTH_MEAN_RADIUS
    xi = Float64(x) * frame.cos_theta - Float64(y) * frame.sin_theta
    eta = Float64(x) * frame.sin_theta + Float64(y) * frame.cos_theta

    lat = frame.lat_center + rad2deg(eta / r_earth)
    mid_lat_rad = deg2rad((lat + frame.lat_center) / 2.0)
    cos_mid = cos(mid_lat_rad)
    if abs(cos_mid) < 1e-10
        error("Polar singularity encountered at latitude $(lat)°.")
    end
    lon = frame.lon_center + rad2deg(xi / (r_earth * cos_mid))
    return (lon, lat)
end

forward_project(proj::RotatedFrame, lon::Real, lat::Real) =
    geographic_to_rotated(lon, lat, proj)

inverse_project(proj::RotatedFrame, x::Real, y::Real) =
    rotated_to_geographic(x, y, proj)

grid_convergence_angle(proj::RotatedFrame, lon::Real, lat::Real) =
    proj.angle_deg

# ==============================================================================
# 4. Standard Projection Constructors via CoordRefSystems
# ==============================================================================

"""
    UTMProjection(;
        zone::Int = 0,
        is_north::Bool = true,
        lon_center::Union{Nothing, Real} = nothing,
        lat_center::Union{Nothing, Real} = nothing,
        rotation_deg::Real = 0.0
    ) -> CoordRefSystemsProjection

Construct a Universal Transverse Mercator (UTM) projection backed by `CoordRefSystems.jl`.
"""
function UTMProjection(;
    zone::Int = 0,
    is_north::Bool = true,
    lon_center::Union{Nothing, Real} = nothing,
    lat_center::Union{Nothing, Real} = nothing,
    rotation_deg::Real = 0.0
)
    utm_zone = if zone <= 0
        l_c = lon_center === nothing ? -62.0 : Float64(lon_center)
        floor(Int, (l_c + 180.0) / 6.0) + 1
    else
        zone
    end

    if utm_zone < 1 || utm_zone > 60
        error("Invalid UTM zone: $(utm_zone). Must be between 1 and 60.")
    end

    crs = is_north ? utmnorth(utm_zone) : utmsouth(utm_zone)

    cx = 0.0
    cy = 0.0
    if lon_center !== nothing && lat_center !== nothing
        pt_ll = LatLon(Float64(lat_center) * °, Float64(lon_center) * °)
        pt_proj = convert(crs, pt_ll)
        cx = Float64(pt_proj.x / m)
        cy = Float64(pt_proj.y / m)
    end

    rot_rad = deg2rad(Float64(rotation_deg))
    return CoordRefSystemsProjection(
        crs, cx, cy, Float64(rotation_deg), cos(rot_rad), sin(rot_rad)
    )
end

"""
    LambertProjection(;
        lat_1::Real,
        lat_2::Real,
        lon_0::Real = 0.0,
        lat_0::Real = 0.0,
        lon_center::Union{Nothing, Real} = nothing,
        lat_center::Union{Nothing, Real} = nothing,
        rotation_deg::Real = 0.0
    ) -> CoordRefSystemsProjection

Construct an ellipsoidal Lambert Conformal Conic (LCC) projection backed by `CoordRefSystems.jl`.
"""
function LambertProjection(;
    lat_1::Real,
    lat_2::Real,
    lon_0::Real = 0.0,
    lat_0::Real = 0.0,
    lon_center::Union{Nothing, Real} = nothing,
    lat_center::Union{Nothing, Real} = nothing,
    rotation_deg::Real = 0.0
)
    p0 = Float64(lat_0) * °
    p1 = Float64(lat_1) * °
    p2 = Float64(lat_2) * °
    l0 = Float64(lon_0) * °

    s = Shift(lonₒ = l0, xₒ = 0.0m, yₒ = 0.0m)
    crs = LambertConic{p0, p1, p2, WGS84Latest, s}

    cx = 0.0
    cy = 0.0
    if lon_center !== nothing && lat_center !== nothing
        pt_ll = LatLon(Float64(lat_center) * °, Float64(lon_center) * °)
        pt_proj = convert(crs, pt_ll)
        cx = Float64(pt_proj.x / m)
        cy = Float64(pt_proj.y / m)
    end

    rot_rad = deg2rad(Float64(rotation_deg))
    return CoordRefSystemsProjection(crs, cx, cy, Float64(rotation_deg), cos(rot_rad), sin(rot_rad))
end

"""
    MercatorProjection(;
        lon_0::Real = 0.0,
        lat_ts::Real = 0.0,
        lon_center::Union{Nothing, Real} = nothing,
        lat_center::Union{Nothing, Real} = nothing,
        rotation_deg::Real = 0.0
    ) -> CoordRefSystemsProjection

Construct a standard conformal Mercator projection backed by `CoordRefSystems.jl`.
"""
function MercatorProjection(;
    lon_0::Real = 0.0,
    lat_ts::Real = 0.0,
    lon_center::Union{Nothing, Real} = nothing,
    lat_center::Union{Nothing, Real} = nothing,
    rotation_deg::Real = 0.0
)
    l0 = Float64(lon_0) * °
    s = Shift(lonₒ = l0, xₒ = 0.0m, yₒ = 0.0m)
    crs = Mercator{WGS84Latest, s}

    cx = 0.0
    cy = 0.0
    if lon_center !== nothing && lat_center !== nothing
        pt_ll = LatLon(Float64(lat_center) * °, Float64(lon_center) * °)
        pt_proj = convert(crs, pt_ll)
        cx = Float64(pt_proj.x / m)
        cy = Float64(pt_proj.y / m)
    end

    rot_rad = deg2rad(Float64(rotation_deg))
    return CoordRefSystemsProjection(crs, cx, cy, Float64(rotation_deg), cos(rot_rad), sin(rot_rad))
end

# ==============================================================================
# 5. Universal Factory Function & Resolvers
# ==============================================================================

"""
    resolve_crs(crs_spec) -> Union{Type{<:CRS}, Nothing}

Resolve a coordinate reference system specification from `CoordRefSystems.jl`.
Supports:
- A `CoordRefSystems.CRS` type directly
- An integer EPSG code (e.g. `32620`)
- An EPSG code string (e.g. `"EPSG:32620"` or `"32620"`)
- An ESRI code string (e.g. `"ESRI:102035"`)
- A Well-Known Text (WKT1 or WKT2) string
"""
function resolve_crs(crs_spec)
    if crs_spec isa Type && crs_spec <: CRSSystem
        return crs_spec
    elseif crs_spec isa Integer
        return CoordRefSystems.get(EPSG{Int(crs_spec)})
    elseif crs_spec isa AbstractString
        str = strip(crs_spec)
        isempty(str) && return nothing
        # EPSG:XXXX or epsg:XXXX
        m_epsg = match(r"^epsg:?\s*([0-9]+)$"i, str)
        if m_epsg !== nothing
            code = parse(Int, m_epsg.captures[1])
            return CoordRefSystems.get(EPSG{code})
        end
        # ESRI:XXXX or esri:XXXX
        m_esri = match(r"^esri:?\s*([0-9]+)$"i, str)
        if m_esri !== nothing
            code = parse(Int, m_esri.captures[1])
            return CoordRefSystems.get(ESRI{code})
        end
        # Pure numeric string e.g. "32620"
        if occursin(r"^[0-9]+$", str)
            code = parse(Int, str)
            return CoordRefSystems.get(EPSG{code})
        end
        # CoordRefSystems resolves WKT1/WKT2 identifiers to its registered CRS types.
        # Do not turn parse failures into an implicit fallback projection.
        c = CoordRefSystems.string2code(str)
        return CoordRefSystems.get(c)
    end
    return nothing
end

"""
    create_projection(;
        projection::Any = "geographic",
        crs::Any = nothing,
        zone::Int = 0,
        lat_1::Union{Nothing, Real} = nothing,
        lat_2::Union{Nothing, Real} = nothing,
        lon_center::Union{Nothing, Real} = nothing,
        lat_center::Union{Nothing, Real} = nothing,
        rotation_deg::Real = 0.0,
        extent_x::Real = 0.0,
        extent_y::Real = 0.0
    ) -> AbstractProjection

Construct an `AbstractProjection` instance from configuration parameters.

# Supported Projections and CRS
- `crs`: Any `CoordRefSystems.CRS` type, integer EPSG code, `"EPSG:XXXX"`, `"ESRI:XXXX"`,
  or WKT string.
- `"geographic"`, `"latlon"`, `"wgs84"`, `"none"`: `GeographicProjection` (unrotated)
  or `RotatedFrame` (if `rotation_deg != 0.0`).
- `"utm"`, `"universal_transverse_mercator"`: Universal Transverse Mercator.
- `"lambert"`, `"lcc"`, `"lambert_conformal_conic"`: Lambert Conformal Conic.
- `"mercator"`: Standard Mercator.
- `"rotated"`, `"oblique"`, `"oblique_mercator"`, `"omerc"`: `RotatedFrame`.
"""
function create_projection(spec::Union{AbstractString, Symbol}; kwargs...)
    str = string(spec)
    clean = lowercase(strip(str))
    if clean in ("lambert", "lcc", "lambert_conformal_conic")
        return LambertProjection(; kwargs...)
    elseif clean in ("utm", "universal_transverse_mercator")
        return UTMProjection(; kwargs...)
    elseif clean in ("mercator",)
        return MercatorProjection(; kwargs...)
    elseif clean in ("geographic", "latlon", "wgs84", "none")
        return GeographicProjection(; kwargs...)
    else
        return create_projection(; crs = str, kwargs...)
    end
end

function _projection_from_resolved_crs(
    resolved; lon_center, lat_center, rotation_deg, extent_x, extent_y
)
    if resolved <: Projected
        cx = 0.0
        cy = 0.0
        if lon_center !== nothing && lat_center !== nothing
            point = convert(resolved, LatLon(Float64(lat_center) * °, Float64(lon_center) * °))
            cx, cy = Float64(point.x / m), Float64(point.y / m)
        end
        angle = Float64(rotation_deg)
        return CoordRefSystemsProjection(
            resolved, cx, cy, angle, cosd(angle), sind(angle))
    elseif resolved <: LatLon
        wgs84 = CoordRefSystems.get(EPSG{4326})
        if resolved === wgs84
            if rotation_deg != 0.0
                lon0 = lon_center === nothing ? 0.0 : Float64(lon_center)
                lat0 = lat_center === nothing ? 0.0 : Float64(lat_center)
                return RotatedFrame(lon0, lat0, rotation_deg, extent_x, extent_y)
            end
            return GeographicProjection()
        end
        rotation_deg == 0.0 || throw(ArgumentError(
            "A rotation cannot be combined with geographic datum CRS $(resolved); " *
            "define the rotation as a separate projected frame."))
        return CoordRefSystemsGeographicProjection(resolved)
    end
    throw(ArgumentError("CoordRefSystems CRS $(resolved) is not a supported 2-D " *
                        "geographic or projected coordinate reference system."))
end

function create_projection(;
    projection::Any = "geographic",
    crs::Any = nothing,
    zone::Int = 0,
    lat_1::Union{Nothing, Real} = nothing,
    lat_2::Union{Nothing, Real} = nothing,
    lon_center::Union{Nothing, Real} = nothing,
    lat_center::Union{Nothing, Real} = nothing,
    rotation_deg::Real = 0.0,
    extent_x::Real = 0.0,
    extent_y::Real = 0.0
)
    # 1. First check explicit CRS argument. An invalid CRS must never fall through to
    #    the default geographic projection.
    if crs !== nothing && crs !== ""
        resolved = try
            resolve_crs(crs)
        catch err
            throw(ArgumentError("Could not resolve CRS $(repr(crs)) with CoordRefSystems: " *
                                sprint(showerror, err)))
        end
        resolved === nothing && throw(ArgumentError(
            "CRS $(repr(crs)) is not supported by CoordRefSystems."))
        return _projection_from_resolved_crs(
            resolved; lon_center, lat_center, rotation_deg, extent_x, extent_y)
    end

    # 2. Named projection aliases are not CRS strings; all other values are resolved
    #    strictly through CoordRefSystems (including WKT and authority identifiers).
    proj_str = string(projection)
    clean_name = replace(lowercase(strip(proj_str)), "-" => "_")
    named_projection = clean_name in (
        "geographic", "latlon", "wgs84", "none", "", "utm",
        "universal_transverse_mercator", "lambert", "lcc",
        "lambert_conformal_conic", "mercator", "transverse_mercator",
        "rotated", "oblique", "oblique_mercator", "omerc")
    resolved_proj = named_projection ? nothing : resolve_crs(projection)
    if resolved_proj !== nothing
        return _projection_from_resolved_crs(
            resolved_proj; lon_center, lat_center, rotation_deg, extent_x, extent_y)
    end

    if clean_name in ("geographic", "latlon", "wgs84", "none", "")
        if rotation_deg != 0.0
            l_c = lon_center === nothing ? 0.0 : Float64(lon_center)
            p_c = lat_center === nothing ? 0.0 : Float64(lat_center)
            return RotatedFrame(l_c, p_c, rotation_deg, extent_x, extent_y)
        end
        return GeographicProjection()
    elseif clean_name in ("utm", "universal_transverse_mercator")
        is_n = lat_center === nothing ? true : Float64(lat_center) >= 0.0
        return UTMProjection(
            zone = zone,
            is_north = is_n,
            lon_center = lon_center,
            lat_center = lat_center,
            rotation_deg = rotation_deg
        )
    elseif clean_name in ("lambert", "lcc", "lambert_conformal_conic")
        l_c = lon_center === nothing ? 0.0 : Float64(lon_center)
        p_c = lat_center === nothing ? 0.0 : Float64(lat_center)
        p1 = lat_1 === nothing || lat_1 == 0.0 ? p_c - 1.5 : Float64(lat_1)
        p2 = lat_2 === nothing || lat_2 == 0.0 ? p_c + 1.5 : Float64(lat_2)
        return LambertProjection(
            lat_1 = p1,
            lat_2 = p2,
            lon_0 = l_c,
            lat_0 = p_c,
            lon_center = lon_center,
            lat_center = lat_center,
            rotation_deg = rotation_deg
        )
    elseif clean_name in ("mercator", "transverse_mercator")
        l_c = lon_center === nothing ? 0.0 : Float64(lon_center)
        p_c = lat_center === nothing ? 0.0 : Float64(lat_center)
        p_ts = lat_1 === nothing || lat_1 == 0.0 ? p_c : Float64(lat_1)
        return MercatorProjection(
            lon_0 = l_c,
            lat_ts = p_ts,
            lon_center = lon_center,
            lat_center = lat_center,
            rotation_deg = rotation_deg
        )
    elseif clean_name in ("rotated", "oblique", "oblique_mercator", "omerc")
        l_c = lon_center === nothing ? 0.0 : Float64(lon_center)
        p_c = lat_center === nothing ? 0.0 : Float64(lat_center)
        return RotatedFrame(l_c, p_c, rotation_deg, extent_x, extent_y)
    else
        error(
            "Unknown projection or CRS '$(projection)'. Supported projections: " *
            "geographic, utm, lambert, mercator, rotated, omerc, EPSG code, ESRI code, or WKT string."
        )
    end
end

# ==============================================================================
# 6. Vector Transformations
# ==============================================================================

"""
    rotate_vector(u_geo::Real, v_geo::Real, angle_deg::Real) -> (Float64, Float64)
    rotate_vector(u_geo::AbstractArray, v_geo::AbstractArray, angle_deg::Real)

Rotate velocity vector components from true geographic orientation (East, North)
into the rotated frame orientation aligned at `angle_deg` counter-clockwise from East:
```math
u' = u \\cos\\theta + v \\sin\\theta
```
```math
v' = -u \\sin\\theta + v \\cos\\theta
```
"""
function rotate_vector(u_geo::Real, v_geo::Real, angle_deg::Real)
    rad = deg2rad(Float64(angle_deg))
    c = cos(rad)
    s = sin(rad)
    u_rot = Float64(u_geo) * c + Float64(v_geo) * s
    v_rot = -Float64(u_geo) * s + Float64(v_geo) * c
    return (u_rot, v_rot)
end

function rotate_vector(
    u_geo::AbstractArray{<:Real},
    v_geo::AbstractArray{<:Real},
    angle_deg::Real
)
    rad = deg2rad(Float64(angle_deg))
    c = cos(rad)
    s = sin(rad)
    u_rot = @. Float64(u_geo) * c + Float64(v_geo) * s
    v_rot = @. -Float64(u_geo) * s + Float64(v_geo) * c
    return (u_rot, v_rot)
end

"""
    unrotate_vector(u_rot::Real, v_rot::Real, angle_deg::Real) -> (Float64, Float64)
    unrotate_vector(u_rot::AbstractArray, v_rot::AbstractArray, angle_deg::Real)

Inverse transformation of velocity vector components from the rotated grid frame
back to true geographic orientation (East, North):
```math
u = u' \\cos\\theta - v' \\sin\\theta
```
```math
v = u' \\sin\\theta + v' \\cos\\theta
```
"""
function unrotate_vector(u_rot::Real, v_rot::Real, angle_deg::Real)
    rad = deg2rad(Float64(angle_deg))
    c = cos(rad)
    s = sin(rad)
    u_geo = Float64(u_rot) * c - Float64(v_rot) * s
    v_geo = Float64(u_rot) * s + Float64(v_rot) * c
    return (u_geo, v_geo)
end

function unrotate_vector(
    u_rot::AbstractArray{<:Real},
    v_rot::AbstractArray{<:Real},
    angle_deg::Real
)
    rad = deg2rad(Float64(angle_deg))
    c = cos(rad)
    s = sin(rad)
    u_geo = @. Float64(u_rot) * c - Float64(v_rot) * s
    v_geo = @. Float64(u_rot) * s + Float64(v_rot) * c
    return (u_geo, v_geo)
end

"""
    rotate_vector_to_grid(proj::AbstractProjection, u_geo, v_geo, lon, lat)

Transform velocity vector components from true geographic orientation to grid-aligned
components taking into account local grid convergence.
"""
function rotate_vector_to_grid(proj::AbstractProjection, u_geo, v_geo, lon, lat)
    gamma = grid_convergence_angle(proj, lon, lat)
    return rotate_vector(u_geo, v_geo, gamma)
end

"""
    rotate_vector_to_geo(proj::AbstractProjection, u_grid, v_grid, lon, lat)

Transform grid-aligned velocity components back to true geographic orientation (East, North).
"""
function rotate_vector_to_geo(proj::AbstractProjection, u_grid, v_grid, lon, lat)
    gamma = grid_convergence_angle(proj, lon, lat)
    return unrotate_vector(u_grid, v_grid, gamma)
end

"""Resolve a declared CRS or dataset `CoordinateSystem` through CoordRefSystems."""
function projection_from_crs(spec; kwargs...)
    if spec isa CoordinateSystem
        candidates = String[]
        !isempty(strip(spec.crs)) && push!(candidates, strip(spec.crs))
        !isnothing(spec.wkt) && !isempty(strip(spec.wkt)) && push!(candidates, strip(spec.wkt))
        !isnothing(spec.proj4) && !isempty(strip(spec.proj4)) && push!(candidates, strip(spec.proj4))
        isempty(candidates) && throw(ArgumentError(
            "CoordinateSystem has no CRS, WKT, or PROJ definition."))
        resolved = map(resolve_crs, candidates)
        all(==(first(resolved)), resolved) || throw(ArgumentError(
            "CoordinateSystem contains conflicting CRS/WKT/PROJ definitions."))
        return create_projection(projection = first(candidates); kwargs...)
    end
    return create_projection(projection = spec; kwargs...)
end

"""Transform one coordinate from `source` CRS coordinates to `target` CRS coordinates."""
function transform_coordinates(
    source::AbstractProjection, target::AbstractProjection, x::Real, y::Real
)
    lon, lat = inverse_project(source, x, y)
    return forward_project(target, lon, lat)
end

"""A stable identity for projection-specific derived geospatial products."""
projection_signature(projection::AbstractProjection) =
    string(typeof(projection), ":", repr(projection))

"""
    transform_vector_dataset(ds, projection; pairs)

Rotate declared geographic east/north vector pairs into `projection`'s local grid frame.
Each pair is `(u_name, v_name, output_u_name, output_v_name)`. Spatial coordinates and
scalar variables are preserved; only listed components are transformed.
"""
function transform_vector_dataset(ds::GeoDataset, projection::AbstractProjection; pairs::Tuple)
    source_projection = projection_from_crs(ds.crs)
    is_planar(source_projection) && throw(ArgumentError(
        "Vector components must be defined on geographic coordinates before rotation; " *
        "dataset CRS is $(repr(ds.crs))."))
    lon, _ = axis(ds, :lon)
    lat, _ = axis(ds, :lat)
    lon_wgs = Matrix{Float64}(undef, length(lon), length(lat))
    lat_wgs = similar(lon_wgs)
    geographic = GeographicProjection()
    for j in eachindex(lat), i in eachindex(lon)
        lon_wgs[i, j], lat_wgs[i, j] = transform_coordinates(
            source_projection, geographic, lon[i], lat[j])
    end

    variables = copy(ds.variables)
    for (u_name, v_name, out_u_name, out_v_name) in pairs
        ukey, vkey = string(u_name), string(v_name)
        haskey(ds.variables, ukey) || throw(ArgumentError(
            "Vector dataset has no variable $(repr(ukey))."))
        haskey(ds.variables, vkey) || throw(ArgumentError(
            "Vector dataset has no variable $(repr(vkey))."))
        u_source, v_source = ds.variables[ukey], ds.variables[vkey]
        size(u_source) == size(v_source) && u_source.dims == v_source.dims ||
            throw(ArgumentError("Vector components $(repr(ukey)) and $(repr(vkey)) must " *
                                "have identical dimensions and sizes."))
        u_grid, v_grid = _rotate_vector_array(
            u_source.data, v_source.data, u_source.dims, lon, lat, lon_wgs, lat_wgs, projection)

        u_attrs, v_attrs = copy(u_source.attrs), copy(v_source.attrs)
        u_attrs["vector_frame"] = "model_grid"
        v_attrs["vector_frame"] = "model_grid"
        u_attrs["source_component"] = ukey
        v_attrs["source_component"] = vkey
        variables[string(out_u_name)] = GeoArray(u_grid, u_source.dims, u_source.crs, u_attrs)
        variables[string(out_v_name)] = GeoArray(v_grid, v_source.dims, v_source.crs, v_attrs)
    end

    attrs = copy(ds.attrs)
    attrs["vector_frame"] = "model_grid"
    attrs["projection_signature"] = projection_signature(projection)
    attrs["transformation"] =
        "Geographic east/north vector components rotated to local model-grid axes"
    return GeoDataset(variables, copy(ds.coords), copy(ds.dims), ds.crs, attrs,
                      ds.backend, ds.source)
end

function _rotate_vector_array(u, v, dims, lon, lat, lon_wgs, lat_wgs, projection)
    permutation = dim_permutation(dims, :lon, :lat)
    remaining = [i for i in eachindex(dims) if !(i in permutation)]
    permutation = vcat(permutation, remaining)
    u_ordered = permutedims(Float64.(Array(u)), permutation)
    v_ordered = permutedims(Float64.(Array(v)), permutation)
    size(u_ordered, 1) == length(lon) && size(u_ordered, 2) == length(lat) ||
        throw(DimensionMismatch("Vector components do not match dataset horizontal axes."))

    u3 = reshape(copy(u_ordered), length(lon), length(lat), :)
    v3 = reshape(copy(v_ordered), length(lon), length(lat), :)
    for k in axes(u3, 3), j in axes(u3, 2), i in axes(u3, 1)
        ue, vn = u3[i, j, k], v3[i, j, k]
        if isfinite(ue) && isfinite(vn)
            u3[i, j, k], v3[i, j, k] = rotate_vector_to_grid(
                projection, ue, vn, lon_wgs[i, j], lat_wgs[i, j])
        end
    end
    shape = size(u_ordered)
    return (permutedims(reshape(u3, shape), invperm(permutation)),
            permutedims(reshape(v3, shape), invperm(permutation)))
end

# ==============================================================================
# 7. Output Reprojection Lattice
# ==============================================================================

"""
    ReprojectionGrid

Cached mapping coordinates for bilinear re-projection from a projected computational
frame to a regular geographic target lattice `(target_lons, target_lats)`.
"""
struct ReprojectionGrid{P <: AbstractProjection}
    projection  :: P
    target_lons :: Vector{Float64}
    target_lats :: Vector{Float64}
    x_coords    :: Matrix{Float64}
    y_coords    :: Matrix{Float64}
    in_bounds   :: BitMatrix
end

"""
    ReprojectionGrid(
        proj::AbstractProjection,
        target_lons::AbstractVector{<:Real},
        target_lats::AbstractVector{<:Real};
        extent_x::Real = Inf,
        extent_y::Real = Inf
    ) -> ReprojectionGrid

Pre-compute coordinate transformation arrays between `(target_lons, target_lats)` and
projected coordinates of `proj`.
"""
function ReprojectionGrid(
    proj::AbstractProjection,
    target_lons::AbstractVector{<:Real},
    target_lats::AbstractVector{<:Real};
    extent_x::Real = Inf,
    extent_y::Real = Inf
)
    tlons = Float64.(collect(target_lons))
    tlats = Float64.(collect(target_lats))
    nx_t = length(tlons)
    ny_t = length(tlats)
    x_coords = Matrix{Float64}(undef, nx_t, ny_t)
    y_coords = Matrix{Float64}(undef, nx_t, ny_t)
    in_bounds = BitMatrix(undef, nx_t, ny_t)

    hx = extent_x isa Real && isfinite(extent_x) && extent_x > 0.0 ? extent_x / 2.0 : Inf
    hy = extent_y isa Real && isfinite(extent_y) && extent_y > 0.0 ? extent_y / 2.0 : Inf

    for j in 1:ny_t, i in 1:nx_t
        x, y = forward_project(proj, tlons[i], tlats[j])
        x_coords[i, j] = x
        y_coords[i, j] = y
        in_bounds[i, j] = (-hx <= x <= hx) && (-hy <= y <= hy)
    end

    return ReprojectionGrid(proj, tlons, tlats, x_coords, y_coords, in_bounds)
end

function ReprojectionGrid(
    frame::RotatedFrame,
    target_lons::AbstractVector{<:Real},
    target_lats::AbstractVector{<:Real}
)
    return ReprojectionGrid(
        frame,
        target_lons,
        target_lats;
        extent_x = frame.extent_x,
        extent_y = frame.extent_y
    )
end

"""
    reproject_to_geographic(
        field_2d::AbstractMatrix{<:Real},
        xs::AbstractVector{<:Real},
        ys::AbstractVector{<:Real},
        grid_map::ReprojectionGrid;
        fill_value::Real = NaN32
    ) -> Matrix{Float32}

Reproject a 2D scalar field from the projected Cartesian coordinate lattice `(xs, ys)`
onto the regular geographic grid defined by `grid_map`.
"""
function reproject_to_geographic(
    field_2d::AbstractMatrix{<:Real},
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    grid_map::ReprojectionGrid;
    fill_value::Real = NaN32
)
    itp = Interpolations.interpolate(
        (Float64.(xs), Float64.(ys)),
        Float32.(field_2d),
        Interpolations.Gridded(Interpolations.Linear())
    )
    itp_ext = Interpolations.extrapolate(itp, Interpolations.Flat())

    nx_t = length(grid_map.target_lons)
    ny_t = length(grid_map.target_lats)
    out = Matrix{Float32}(undef, nx_t, ny_t)

    for j in 1:ny_t, i in 1:nx_t
        if grid_map.in_bounds[i, j]
            out[i, j] = itp_ext(grid_map.x_coords[i, j], grid_map.y_coords[i, j])
        else
            out[i, j] = Float32(fill_value)
        end
    end
    return out
end

"""
    reproject_to_geographic(
        field_3d::AbstractArray{<:Real, 3},
        xs::AbstractVector{<:Real},
        ys::AbstractVector{<:Real},
        grid_map::ReprojectionGrid;
        fill_value::Real = NaN32
    ) -> Array{Float32, 3}

Reproject each horizontal layer of a 3D field from the projected lattice `(xs, ys)`
onto the geographic grid of `grid_map`.
"""
function reproject_to_geographic(
    field_3d::AbstractArray{<:Real, 3},
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    grid_map::ReprojectionGrid;
    fill_value::Real = NaN32
)
    nx_t = length(grid_map.target_lons)
    ny_t = length(grid_map.target_lats)
    nz = size(field_3d, 3)
    out = Array{Float32, 3}(undef, nx_t, ny_t, nz)

    for k in 1:nz
        out[:, :, k] = reproject_to_geographic(
            field_3d[:, :, k], xs, ys, grid_map; fill_value = fill_value
        )
    end
    return out
end

# ==============================================================================
# Geodetic UTM & Coordinate System Transformations
# ==============================================================================

"""
    utm_to_lonlat(easting::Real, northing::Real; zone::Int=20, is_km::Bool=true, northern::Bool=true)
    -> Tuple{Float64, Float64}

Converts Universal Transverse Mercator (UTM) coordinates (in km or meters) to WGS84
geographic coordinates `(lon, lat)` in decimal degrees using `CoordRefSystems.jl`.

# Mathematical Formulation
Let \$(x, y)\$ be the projected UTM Cartesian coordinates in meters. The geodetic
coordinates \$(lon, lat)\$ on the WGS84 reference ellipsoid are obtained via the inverse
Transverse Mercator mapping:
```math
(\\lambda, \\phi) = T_{\\text{UTM}}^{-1}(x, y; \\text{zone}, \\text{datum}=\\text{WGS84})
```

# Arguments
- `easting::Real`: Projected X coordinate (Easting).
- `northing::Real`: Projected Y coordinate (Northing).
- `zone::Int`: UTM Longitudinal Zone (\$1 \\le \\text{zone} \\le 60\$, default 20).
- `is_km::Bool`: Whether inputs are in kilometers (`true`, default) or meters (`false`).
- `northern::Bool`: Whether the coordinate is in the Northern Hemisphere (default `true`).

# Returns
- `Tuple{Float64, Float64}`: Geographic `(lon, lat)` in decimal degrees.
"""
function utm_to_lonlat(
    easting::Real,
    northing::Real;
    zone::Int = 20,
    is_km::Bool = true,
    northern::Bool = true
)::Tuple{Float64, Float64}
    scale_m = is_km ? 1000.0 : 1.0
    x_m = Float64(easting) * scale_m
    y_m = Float64(northing) * scale_m
    crs = northern ? utmnorth(zone) : utmsouth(zone)
    pt = crs(x_m * m, y_m * m)
    ll = convert(LatLon, pt)
    lon_deg = Float64(ll.lon / °)
    lat_deg = Float64(ll.lat / °)
    return (lon_deg, lat_deg)
end

"""
    lonlat_to_utm(lon::Real, lat::Real; zone::Int=20, is_km::Bool=true, northern::Bool=true)
    -> Tuple{Float64, Float64}

Converts WGS84 geographic coordinates `(lon, lat)` in decimal degrees to Universal Transverse
Mercator (UTM) coordinates (in km or meters) using `CoordRefSystems.jl`.

# Mathematical Formulation
Let \$(lon, lat)\$ be the geodetic longitude and latitude in degrees. The projected
Cartesian coordinates \$(x, y)\$ on the Transverse Mercator plane with central meridian
\$\\lambda_0 = 6^\\circ \\cdot (\\text{zone} - 1) - 180^\\circ + 3^\\circ\$ are given by:
```math
(x, y) = T_{\\text{UTM}}(\\lambda, \\phi; \\text{zone}, \\text{datum}=\\text{WGS84})
```

# Arguments
- `lon::Real`: Longitude in decimal degrees ([-180, 180]).
- `lat::Real`: Latitude in decimal degrees ([-90, 90]).
- `zone::Int`: UTM Longitudinal Zone (\$1 \\le \\text{zone} \\le 60\$, default 20).
- `is_km::Bool`: Whether output should be scaled to kilometers (`true`, default) or meters (`false`).
- `northern::Bool`: Whether the coordinate is in the Northern Hemisphere (default `true`).

# Returns
- `Tuple{Float64, Float64}`: Projected Easting and Northing coordinates.
"""
function lonlat_to_utm(
    lon::Real,
    lat::Real;
    zone::Int = 20,
    is_km::Bool = true,
    northern::Bool = true
)::Tuple{Float64, Float64}
    crs = northern ? utmnorth(zone) : utmsouth(zone)
    ll = LatLon(Float64(lat) * °, Float64(lon) * °)
    pt = convert(crs, ll)
    x_m = Float64(pt.x / m)
    y_m = Float64(pt.y / m)
    scale_out = is_km ? 0.001 : 1.0
    return (x_m * scale_out, y_m * scale_out)
end


"""
    extract_wkt(au)::Union{Nothing, String}

Safely extracts a Well-Known Text (WKT) or projection definition string from an areal
units container, NamedTuple, or Dictionary.

# Inputs
- `au`: Areal units object, NamedTuple, or AbstractDict.

# Outputs
- `Union{Nothing, String}`: WKT or projection string if present, otherwise `nothing`.
"""
function extract_wkt(au)::Union{Nothing, String}
    if isnothing(au)
        return nothing
    end
    for k in (:wkt, :crs, :proj, :projection, :coord_sys)
        if hasproperty(au, k)
            v = getproperty(au, k)
            v isa AbstractString && !isempty(v) && return string(v)
        elseif au isa AbstractDict && haskey(au, k)
            v = au[k]
            v isa AbstractString && !isempty(v) && return string(v)
        end
    end
    return nothing
end

"""
    is_wgs84_wkt(wkt::Union{Nothing, AbstractString})::Bool

Infers whether a Well-Known Text (WKT) definition represents the standard WGS84
geographic coordinate system.

# Inputs
- `wkt::Union{Nothing, AbstractString}`: WKT or projection definition string.

# Outputs
- `Bool`: `true` if WKT matches WGS84, `false` otherwise.
"""
function is_wgs84_wkt(wkt::Union{Nothing, AbstractString})::Bool
    if isnothing(wkt) || isempty(wkt)
        return false
    end
    return resolve_crs(wkt) === CoordRefSystems.get(EPSG{4326})
end

"""
    CoordinateTransformer

A strict transformer from coordinates in one declared CRS to WGS84 longitude/latitude.
The CRS is represented by a Tethys projection resolved through CoordRefSystems; coordinates
are never classified by their numeric ranges.
"""
struct CoordinateTransformer{P <: AbstractProjection}
    projection::P
    crs_spec::String
end

"""
    build_coordinate_transformer(points; crs=nothing, wkt=nothing, is_geo=nothing)

Build a transformer from an explicitly declared CRS. `points` is retained for API
compatibility and must contain coordinate pairs, but its values are never used to infer a CRS.
Set `is_geo=true` only as an explicit declaration when no CRS string is available.
"""
function build_coordinate_transformer(
    points;
    crs = nothing,
    wkt::Union{Nothing, AbstractString} = nothing,
    is_geo::Union{Nothing, Bool} = nothing,
    lon_center::Union{Nothing, Real} = nothing,
    lat_center::Union{Nothing, Real} = nothing
)
    spec = !isnothing(wkt) && !isempty(strip(wkt)) ? String(wkt) : crs
    if isnothing(spec) && is_geo === true
        spec = "EPSG:4326"
    end
    isnothing(spec) && throw(ArgumentError(
        "A source CRS is required. Supply `crs` or `wkt`; coordinate ranges are not used " *
        "to infer a coordinate system."))

    projection = create_projection(
        projection = spec,
        lon_center = lon_center,
        lat_center = lat_center
    )
    if is_geo === true && is_planar(projection)
        throw(ArgumentError("is_geo=true conflicts with the declared projected CRS $(repr(spec))."))
    elseif is_geo === false && !is_planar(projection)
        throw(ArgumentError("is_geo=false conflicts with the declared geographic CRS $(repr(spec))."))
    end
    for point in points
        length(point) >= 2 || throw(ArgumentError("Every coordinate must contain at least x and y."))
    end
    return CoordinateTransformer(projection, string(spec))
end

"""
    transform_point(tf::CoordinateTransformer, pt)::Tuple{Float64, Float64}

Transforms a 2D coordinate pair `(x, y)` into WGS84 geographic coordinates `(lon, lat)`
using transformer `tf`.

# Inputs
- `tf::CoordinateTransformer`: Coordinate transformation engine.
- `pt`: 2D point tuple or vector `(x, y)`.

# Outputs
- `Tuple{Float64, Float64}`: Geographic coordinates `(lon, lat)` in decimal degrees.
"""
function transform_point(tf::CoordinateTransformer, pt)::Tuple{Float64, Float64}
    length(pt) >= 2 || throw(ArgumentError("A coordinate must contain at least x and y."))
    return inverse_project(tf.projection, Float64(pt[1]), Float64(pt[2]))
end

"""
    transform_polygon(tf::CoordinateTransformer, poly)

Transforms all vertices of polygon `poly` into WGS84 geographic coordinates.

# Inputs
- `tf::CoordinateTransformer`: Coordinate transformation engine.
- `poly`: Collection of 2D vertex coordinate pairs.

# Outputs
- `Vector{Tuple{Float64, Float64}}`: Array of transformed geographic coordinates.
"""
function transform_polygon(tf::CoordinateTransformer, poly)
    return [
        transform_point(tf, pt)
        for pt in poly
        if length(pt) >= 2 && !isnan(pt[1]) && !isnan(pt[2])
    ]
end

end # module CRS
