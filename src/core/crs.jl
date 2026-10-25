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

export
    AbstractProjection,
    GeographicProjection,
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
    is_geographic_coordinates,
    extract_wkt,
    is_wgs84_wkt,
    CoordinateTransformer,
    build_coordinate_transformer,
    transform_point,
    transform_polygon

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
        # WKT or named CRS in CoordRefSystems
        try
            c = CoordRefSystems.string2code(str)
            return CoordRefSystems.get(c)
        catch
            return nothing
        end
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
    # 1. First check explicit crs argument
    if crs !== nothing && crs !== ""
        resolved = resolve_crs(crs)
        if resolved !== nothing
            if resolved <: Projected
                cx = 0.0
                cy = 0.0
                if lon_center !== nothing && lat_center !== nothing
                    pt_ll = LatLon(Float64(lat_center) * °, Float64(lon_center) * °)
                    pt_proj = convert(resolved, pt_ll)
                    cx = Float64(pt_proj.x / m)
                    cy = Float64(pt_proj.y / m)
                end
                rot_rad = deg2rad(Float64(rotation_deg))
                return CoordRefSystemsProjection(
                    resolved, cx, cy, Float64(rotation_deg), cos(rot_rad), sin(rot_rad)
                )
            else
                if rotation_deg != 0.0
                    l_c = lon_center === nothing ? 0.0 : Float64(lon_center)
                    p_c = lat_center === nothing ? 0.0 : Float64(lat_center)
                    return RotatedFrame(l_c, p_c, rotation_deg, extent_x, extent_y)
                end
                return GeographicProjection()
            end
        end
    end

    # 2. If projection is a CRS type or integer EPSG code or WKT string
    resolved_proj = resolve_crs(projection)
    if resolved_proj !== nothing
        if resolved_proj <: Projected
            cx = 0.0
            cy = 0.0
            if lon_center !== nothing && lat_center !== nothing
                pt_ll = LatLon(Float64(lat_center) * °, Float64(lon_center) * °)
                pt_proj = convert(resolved_proj, pt_ll)
                cx = Float64(pt_proj.x / m)
                cy = Float64(pt_proj.y / m)
            end
            rot_rad = deg2rad(Float64(rotation_deg))
            return CoordRefSystemsProjection(
                resolved_proj, cx, cy, Float64(rotation_deg), cos(rot_rad), sin(rot_rad)
            )
        else
            if rotation_deg != 0.0
                l_c = lon_center === nothing ? 0.0 : Float64(lon_center)
                p_c = lat_center === nothing ? 0.0 : Float64(lat_center)
                return RotatedFrame(l_c, p_c, rotation_deg, extent_x, extent_y)
            end
            return GeographicProjection()
        end
    end

    proj_str = string(projection)
    clean_name = lowercase(strip(proj_str))
    clean_name = replace(clean_name, "-" => "_")

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
    is_geographic_coordinates(coords)::Bool

Infers whether spatial coordinates are in geographic `(lon, lat)` space versus abstract
planar Cartesian `(x, y)` space.

# Mathematical Criteria
1. Hard WGS84 bounding box: Longitudes must strictly reside within `[-180, 180]`,
   and Latitudes within `[-90, 90]`.
2. Planar origin check: Non-negative coordinates starting near `(0, 0)` with synthetic
   extents (e.g. `[0, 100]`, `[0, 10]`, `[0, 1]`) are classified as planar.
3. Marine regional domain checks (Scotian Shelf, Atlantic Canada, Pacific, Europe).

# Inputs
- `coords`: Collection of 2D coordinate pairs `(x, y)`.

# Outputs
- `Bool`: `true` if coordinates represent geographic WGS84 degrees, `false` otherwise.
"""
function is_geographic_coordinates(coords)::Bool
    if isempty(coords)
        return false
    end
    xs = Float64[]
    ys = Float64[]
    for pt in coords
        if length(pt) >= 2 && !isnan(pt[1]) && !isnan(pt[2])
            push!(xs, float(pt[1]))
            push!(ys, float(pt[2]))
        end
    end
    if isempty(xs) || isempty(ys)
        return false
    end
    min_x, max_x = minimum(xs), maximum(xs)
    min_y, max_y = minimum(ys), maximum(ys)

    # 1. Hard limits for WGS84 Geographic degrees
    if min_y < -90.0 || max_y > 90.0 || min_x < -180.0 || max_x > 180.0
        return false
    end

    # 2. Planar Cartesian origin check: non-negative coordinates starting near (0, 0)
    if min_x >= 0.0 && min_y >= 0.0 && (min_x < 5.0 || min_y < 5.0) &&
       (max_x <= 100.0 && max_y <= 100.0)
        return false
    end

    # 3. Known marine fisheries regions (Atlantic Canada, Scotian Shelf, Pacific, Europe)
    if (min_x <= -20.0 && max_x <= -10.0 && min_y >= 30.0 && max_y <= 85.0) ||
       (min_x >= 100.0 && max_x <= 180.0 && min_y >= -50.0 && max_y <= 70.0) ||
       (min_x >= -180.0 && max_x <= -50.0 && min_y >= -60.0 && max_y <= 75.0)
        return true
    end

    # 4. Standard degrees span check
    if (max_x - min_x > 0.1 || max_y - min_y > 0.1) &&
       (min_x < -5.0 || max_x > 5.0 || min_y > 15.0 || min_y < -15.0)
        return true
    end

    return false
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
    w = uppercase(string(wkt))
    return contains(w, "4326") ||
           contains(w, "WGS 84") ||
           contains(w, "WGS_1984") ||
           contains(w, "GCS_WGS_1984") ||
           contains(w, "GEOGCS[\"WGS") ||
           contains(w, "+PROJ=LONGLAT") ||
           contains(w, "+PROJ=LATLONG")
end

"""
    CoordinateTransformer

Encapsulates coordinate normalization and transformation from raw input space
(geographic degrees, CoordRefSystems projection, UTM, or planar Cartesian) into WGS84
geographic coordinates.

# Fields
- `mode::Symbol`: Transformation mode (`:geographic`, `:crs`, `:utm`, `:planar`).
- `is_geo::Bool`: Whether the source coordinate system is geographic degrees.
- `crs_type::Any`: CoordRefSystems CRS type or instance.
- `utm_zone::Int`: UTM Longitudinal Zone (1 <= zone <= 60).
- `utm_is_km::Bool`: Whether UTM inputs are in kilometers (`true`) or meters (`false`).
- `northern::Bool`: Whether coordinate space resides in the Northern Hemisphere.
- `scale::Float64`: Isometric scaling factor for planar coordinates.
- `x_mean::Float64`: Center Easting / X offset for planar scaling.
- `y_mean::Float64`: Center Northing / Y offset for planar scaling.
- `lon_center::Float64`: Domain center longitude in degrees East.
- `lat_center::Float64`: Domain center latitude in degrees North.
- `wkt::Union{Nothing, String}`: Associated WKT definition string.
"""
struct CoordinateTransformer
    mode::Symbol
    is_geo::Bool
    crs_type::Any
    utm_zone::Int
    utm_is_km::Bool
    northern::Bool
    scale::Float64
    x_mean::Float64
    y_mean::Float64
    lon_center::Float64
    lat_center::Float64
    wkt::Union{Nothing, String}
end

"""
    build_coordinate_transformer(all_points; wkt=nothing, is_geo=nothing,
                                 lon_center=-160.0, lat_center=0.0) -> CoordinateTransformer

Infers coordinate system properties and constructs a `CoordinateTransformer` instance.

# Inputs
- `all_points`: Collection of 2D vertex coordinates or centroids.
- `wkt`: Optional WKT definition string.
- `is_geo`: Optional boolean override for geographic coordinates.
- `lon_center::Float64`: Default center longitude for planar projection.
- `lat_center::Float64`: Default center latitude for planar projection.

# Outputs
- `CoordinateTransformer`: Configured coordinate transformation engine.
"""
function build_coordinate_transformer(
    all_points;
    wkt::Union{Nothing, AbstractString} = nothing,
    is_geo::Union{Nothing, Bool} = nothing,
    lon_center::Float64 = -160.0,
    lat_center::Float64 = 0.0
)::CoordinateTransformer
    wkt_str = !isnothing(wkt) ? string(wkt) : nothing
    has_wgs84_wkt = is_wgs84_wkt(wkt_str)

    # 1. Explicit geographic or WGS84 WKT
    if is_geo === true || has_wgs84_wkt
        return CoordinateTransformer(
            :geographic, true, nothing, 20, false, true, 1.0, 0.0, 0.0, 0.0, 0.0, wkt_str
        )
    end

    # Extract all valid coordinates to determine spatial envelope
    xs = Float64[]
    ys = Float64[]
    for pt in all_points
        if length(pt) >= 2 && !isnan(pt[1]) && !isnan(pt[2])
            push!(xs, float(pt[1]))
            push!(ys, float(pt[2]))
        end
    end

    min_x = !isempty(xs) ? minimum(xs) : 0.0
    max_x = !isempty(xs) ? maximum(xs) : 0.0
    min_y = !isempty(ys) ? minimum(ys) : 0.0
    max_y = !isempty(ys) ? maximum(ys) : 0.0

    # 2. Check for UTM signatures or EPSG codes in WKT / metadata
    is_utm_wkt = !isnothing(wkt_str) && contains(lowercase(wkt_str), "utm")
    utm_zone = 20
    is_north = true
    if is_utm_wkt
        m_zone = match(r"zone\s*=?\s*(\d+)", lowercase(wkt_str))
        if m_zone !== nothing
            utm_zone = parse(Int, m_zone.captures[1])
        end
        if contains(lowercase(wkt_str), "south")
            is_north = false
        end
    end

    # Check EPSG code directly if present in WKT / CRS
    if !isnothing(wkt_str)
        m_epsg = match(r"epsg[:\s]+(\d+)"i, wkt_str)
        if m_epsg !== nothing
            code = parse(Int, m_epsg.captures[1])
            if 32601 <= code <= 32660
                utm_zone = code - 32600
                is_north = true
                is_utm_wkt = true
            elseif 32701 <= code <= 32760
                utm_zone = code - 32700
                is_north = false
                is_utm_wkt = true
            end
        end
    end

    is_utm_km = (min_x >= 50.0 && max_x <= 1500.0 && min_y >= 1000.0 && max_y <= 9000.0)
    is_utm_m = (min_x >= 50000.0 && min_y >= 1000000.0)

    if is_utm_wkt || is_utm_km || is_utm_m
        is_km = is_utm_km ||
            (!is_utm_m && !contains(lowercase(something(wkt_str, "")), "units=m"))
        crs = is_north ? utmnorth(utm_zone) : utmsouth(utm_zone)
        return CoordinateTransformer(
            :utm, true, crs, utm_zone, is_km, is_north, 1.0, 0.0, 0.0, 0.0, 0.0, wkt_str
        )
    end

    geo_flag = if is_geo !== nothing
        is_geo
    else
        is_geographic_coordinates(all_points)
    end

    if geo_flag
        return CoordinateTransformer(
            :geographic, true, nothing, 20, true, true, 1.0, 0.0, 0.0, 0.0, 0.0, wkt_str
        )
    end

    if isempty(xs) || isempty(ys)
        return CoordinateTransformer(
            :planar, false, nothing, 20, true, true,
            1.0, 0.0, 0.0, lon_center, lat_center, wkt_str
        )
    end

    dx = max_x - min_x
    dy = max_y - min_y
    x_bar = (min_x + max_x) / 2.0
    y_bar = (min_y + max_y) / 2.0
    s = dy > 1e-9 ? 1.0 / dy : (dx > 1e-9 ? 1.0 / dx : 1.0)

    return CoordinateTransformer(
        :planar, false, nothing, 20, true, true,
        s, x_bar, y_bar, lon_center, lat_center, wkt_str
    )
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
    x, y = float(pt[1]), float(pt[2])
    # 1. If coordinate is already in standard geographic WGS84 range
    if -180.0 <= x <= 180.0 && -90.0 <= y <= 90.0
        return (x, y)
    end

    # 2. CoordRefSystems transformation mode
    if tf.mode == :utm || tf.mode == :crs
        if !isnothing(tf.crs_type)
            try
                scale_m = tf.utm_is_km ? 1000.0 : 1.0
                pt_crs = tf.crs_type((x * scale_m) * m, (y * scale_m) * m)
                ll = convert(LatLon, pt_crs)
                lon = Float64(ll.lon / °)
                lat = Float64(ll.lat / °)
                if !isnan(lon) && !isnan(lat) && isfinite(lon) && isfinite(lat)
                    return (lon, lat)
                end
            catch
            end
        end
        return utm_to_lonlat(
            x, y; zone = tf.utm_zone, is_km = tf.utm_is_km, northern = tf.northern
        )
    end

    # 3. If explicit geographic mode
    if tf.is_geo
        return (x, y)
    end

    # 4. Fallback planar isometric projection
    lon = (x - tf.x_mean) * tf.scale + tf.lon_center
    lat = (y - tf.y_mean) * tf.scale + tf.lat_center
    return (lon, lat)
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
