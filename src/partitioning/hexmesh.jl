# -----------------------------------------------------------------------------
# hexmesh.jl -- (GeoPartitioning) extracted verbatim from TethysBayes src/partitioning.jl
# lines 2368-2779 (planar hexagonal mesh projection and point mapping).
# ------------------------------------------------------------------------------
"""
    lonlat_to_xy_km(lon, lat; center_lon=nothing, center_lat=nothing, crs=nothing, datum=WGS84Latest)
    -> Tuple{Float64, Float64}

Projects geographic coordinates (longitude, latitude in decimal degrees) into a planar
Cartesian coordinate space in kilometers, optionally relative to a local tangent origin.

# Mathematical Formulation
When `crs` is supplied and `CoordRefSystems` is available, coordinates are converted
via geodesic projection. Otherwise, a local equirectangular tangent plane approximation
is computed:
```math
x = R \\cdot (\\lambda - \\lambda_0) \\cdot \\cos(\\phi_0) \\cdot \\frac{\\pi}{180^\\circ}
```
```math
y = R \\cdot (\\phi - \\phi_0) \\cdot \\frac{\\pi}{180^\\circ}
```
where \$R = 6371.0\\text{ km}\$ is Earth's mean radius, \$(\\lambda_0, \\phi_0)\$ is the
tangent projection origin (`center_lon`, `center_lat`), and \$(\\lambda, \\phi)\$ are
the input longitude and latitude.

# Arguments
- `lon::Real`: Longitude in decimal degrees.
- `lat::Real`: Latitude in decimal degrees.
- `center_lon`: Tangent plane projection center longitude (default `nothing`).
- `center_lat`: Tangent plane projection center latitude (default `nothing`).
- `crs`: Optional target Coordinate Reference System from `CoordRefSystems`.
- `datum`: Geographic datum (default `WGS84Latest`).

# Returns
- `Tuple{Float64, Float64}`: Planar coordinates `(x_km, y_km)` in kilometers.
"""
function lonlat_to_xy_km(
    lon::Real, lat::Real;
    center_lon = nothing,
    center_lat = nothing,
    crs = nothing,
    datum = WGS84Latest
)::Tuple{Float64, Float64}
    if !isnothing(crs) && _HAS_COORDSYS
        try
            source_pt = LatLon{datum}(lat, lon)
            proj_pt = convert(crs, source_pt)

            x_km = Float64(ustrip(u"km", proj_pt.x))
            y_km = Float64(ustrip(u"km", proj_pt.y))

            # If a local custom center is provided, subtract it to get relative offsets
            if !isnothing(center_lon) && !isnothing(center_lat)
                center_pt = LatLon{datum}(center_lat, center_lon)
                proj_center = convert(crs, center_pt)
                x_km -= Float64(ustrip(u"km", proj_center.x))
                y_km -= Float64(ustrip(u"km", proj_center.y))
            end

            return (x_km, y_km)
        catch
        end
    end
    
    # Fallback to local tangent plane (requires a center to function, defaults to 0.0)
    c_lon = isnothing(center_lon) ? 0.0 : Float64(center_lon)
    c_lat = isnothing(center_lat) ? 0.0 : Float64(center_lat)
    
    R  = 6371.0
    ϕ0 = c_lat * (π / 180.0)
    x  = (Float64(lon) - c_lon) * cos(ϕ0) * (π / 180.0) * R
    y  = (Float64(lat) - c_lat) * (π / 180.0) * R
    return (x, y)
end

"""
    xy_km_to_lonlat(x_km, y_km; center_lon=nothing, center_lat=nothing, crs=nothing, datum=WGS84Latest)
    -> Tuple{Float64, Float64}

Inverts planar Cartesian coordinates in kilometers back to geographic coordinates
(longitude, latitude in decimal degrees).

# Mathematical Formulation
For the local equirectangular tangent plane fallback:
```math
\\lambda = \\lambda_0 + \\frac{x}{R \\cdot \\cos(\\phi_0) \\cdot (\\pi / 180^\\circ)}
```
```math
\\phi = \\phi_0 + \\frac{y}{R \\cdot (\\pi / 180^\\circ)}
```
where \$R = 6371.0\\text{ km}\$ is Earth's mean radius and \$(\\lambda_0, \\phi_0)\$ is
the tangent plane projection origin (`center_lon`, `center_lat`).

# Arguments
- `x::Real`: Planar x-coordinate in kilometers.
- `y::Real`: Planar y-coordinate in kilometers.
- `center_lon`: Tangent plane projection center longitude (default `nothing`).
- `center_lat`: Tangent plane projection center latitude (default `nothing`).
- `crs`: Optional target Coordinate Reference System from `CoordRefSystems`.
- `datum`: Geographic datum (default `WGS84Latest`).

# Returns
- `Tuple{Float64, Float64}`: Geographic coordinates `(lon, lat)` in decimal degrees.
"""
function xy_km_to_lonlat(
    x::Real, y::Real;
    center_lon = nothing,
    center_lat = nothing,
    crs = nothing,
    datum = WGS84Latest
)::Tuple{Float64, Float64}
    if !isnothing(crs) && _HAS_COORDSYS
        try
            abs_x = x * 1.0u"km"
            abs_y = y * 1.0u"km"

            # If a local custom center was used, add it back to get absolute CRS coordinates
            if !isnothing(center_lon) && !isnothing(center_lat)
                center_pt = LatLon{datum}(center_lat, center_lon)
                proj_center = convert(crs, center_pt)
                abs_x += proj_center.x
                abs_y += proj_center.y
            end

            proj_pt = crs(abs_x, abs_y)
            lonlat_pt = convert(LatLon{datum}, proj_pt)

            out_lon = Float64(ustrip(u"°", lonlat_pt.lon))
            out_lat = Float64(ustrip(u"°", lonlat_pt.lat))

            return (out_lon, out_lat)
        catch
        end
    end
    
    # Fallback local tangent plane
    c_lon = isnothing(center_lon) ? 0.0 : Float64(center_lon)
    c_lat = isnothing(center_lat) ? 0.0 : Float64(center_lat)
    
    R  = 6371.0
    ϕ0 = c_lat * (π / 180.0)
    lon = c_lon + (Float64(x) / (R * cos(ϕ0))) * (180.0 / π)
    lat = c_lat + (Float64(y) / R) * (180.0 / π)
    return (lon, lat)
end

# ── Internal helpers ───────────────────────────────────────────────────────────

"""
    _to_decimal_year(d) -> Float64

Convert a `Date` or `DateTime` to decimal year, accounting for leap years:

    decimal_year = year + (dayofyear − 1) / days_in_year
"""
function _to_decimal_year(d::Union{Date, DateTime})::Float64
    yr         = Dates.year(d)
    doy        = Float64(Dates.dayofyear(d))
    days_in_yr = Dates.isleapyear(yr) ? 366.0 : 365.0
    return Float64(yr) + (doy - 1.0) / days_in_yr
end

"""
    _safe_mean(v) -> Float64

Mean of `v` after stripping `Missing` and non-finite values.
Returns `NaN` when no valid values remain.
"""
function _safe_mean(v)::Float64
    vals = collect(skipmissing(v))
    vals = filter(x -> (x isa Real) && isfinite(x), vals)
    return isempty(vals) ? NaN : mean(vals)
end


# ── Planar hexagonal mesh ──────────────────────────────────────────────────────

"""
    build_hex_mesh(lon_vec, lat_vec; radius_km=5.0, crs=nothing, datum=WGS84Latest)
    -> NamedTuple

Construct a regular hexagonal tessellation in planar km coordinates from
observation lon/lat points, then back-project polygon vertices and centroids
to geographic coordinates for reporting and visualisation.

# Hex geometry (flat-top orientation)
For circumradius r (km = side length for a regular hexagon):
- Column spacing: dx = √3 r
- Row spacing:    dy = 1.5 r
- Offset row grid: odd rows shift by dx/2.
- Exact area:     A = (3√3/2) r²  km²

# Adjacency
W_ij = 1 when planar centroid distance < √3 r × 1.05 (5 % snap tolerance).
Symmetrised: W = max(W, Wᵀ).

# Arguments
- `lon_vec, lat_vec`: Geographic coordinates in decimal degrees.
- `radius_km`: Hexagon circumradius = side length in km (default 5.0).
- `crs`: Target Coordinate Reference System (default nothing, uses local tangent plane).
- `datum`: The geographic datum (default: `WGS84Latest`).

# Returns
`NamedTuple`:
- `centroids_km, centroids_lonlat`: Centroid coordinates in km and degrees.
- `polygons_km, polygons_lonlat`: Closed vertex rings per hex cell.
- `n_units::Int`, `W::SparseMatrixCSC`, `radius_km::Float64`.
- `areas_km2::Vector{Float64}`: Identical exact hex area per cell.
- `center_lon, center_lat`: Projection origin in degrees.
"""
function build_hex_mesh(
    bbox::Tuple{Real, Real, Real, Real};
    radius_km::Real = 5.0,
    cell_size_km::Union{Nothing, Real} = nothing,
    kwargs...
)::NamedTuple
    r = cell_size_km !== nothing ? Float64(cell_size_km) / sqrt(3.0) : Float64(radius_km)
    lons = [bbox[1], bbox[2], bbox[1], bbox[2]]
    lats = [bbox[3], bbox[3], bbox[4], bbox[4]]
    return build_hex_mesh(lons, lats; radius_km = r, kwargs...)
end

function build_hex_mesh(
    lon_vec::AbstractVector{<:Real},
    lat_vec::AbstractVector{<:Real};
    radius_km::Real = 5.0,
    cell_size_km::Union{Nothing, Real} = nothing,
    crs = nothing,
    datum = WGS84Latest
)::NamedTuple
    r_val = cell_size_km !== nothing ? Float64(cell_size_km) / sqrt(3.0) : Float64(radius_km)

    center_lon = mean(lon_vec)
    center_lat = mean(lat_vec)

    # Calculate planar coordinates directly
    pts_km = [lonlat_to_xy_km(Float64(lon), Float64(lat);
                  center_lon=center_lon, center_lat=center_lat, 
                  crs=crs, datum=datum)
              for (lon, lat) in zip(lon_vec, lat_vec)]

    r  = r_val
    dx = sqrt(3.0) * r
    dy = 1.5 * r

    # Determine planar domain bounds (expanded by r to ensure complete boundary coverage)
    x_coords = [p[1] for p in pts_km]
    y_coords = [p[2] for p in pts_km]
    x_min, x_max = extrema(x_coords)
    y_min, y_max = extrema(y_coords)

    x_min -= r
    x_max += r
    y_min -= r
    y_max += r

    row_min = floor(Int, y_min / dy)
    row_max = ceil(Int, y_max / dy)

    centroids_km = Tuple{Float64, Float64}[]
    for row in row_min:row_max
        yk   = row * dy
        xoff = isodd(row) ? (dx / 2.0) : 0.0
        col_min = floor(Int, (x_min - xoff) / dx)
        col_max = ceil(Int, (x_max - xoff) / dx)
        for col in col_min:col_max
            xk = col * dx + xoff
            push!(centroids_km, (xk, yk))
        end
    end

    sort!(centroids_km, by = c -> (c[2], c[1]))
    S = length(centroids_km)

    polygons_km      = Vector{Vector{Tuple{Float64, Float64}}}(undef, S)
    polygons_lonlat  = Vector{Vector{Tuple{Float64, Float64}}}(undef, S)
    centroids_lonlat = Vector{Tuple{Float64, Float64}}(undef, S)

    # Flat-top hexagon: vertices at 30°, 90°, 150°, 210°, 270°, 330°
    hex_angles = (30.0 .+ 60.0 .* (0:5)) .* (π / 180.0)

    for (i, (cx, cy)) in enumerate(centroids_km)
        verts_km = [(cx + r * cos(a), cy + r * sin(a)) for a in hex_angles]
        push!(verts_km, verts_km[1]) # Close the polygon
        
        polygons_km[i]      = verts_km
        polygons_lonlat[i]  = [xy_km_to_lonlat(v[1], v[2];
                                    center_lon=center_lon, center_lat=center_lat,
                                    crs=crs, datum=datum)
                                for v in verts_km]
        
        centroids_lonlat[i] = xy_km_to_lonlat(cx, cy;
                                    center_lon=center_lon, center_lat=center_lat,
                                    crs=crs, datum=datum)
    end

    # Efficient matrix allocation for KDTree
    c_mat = Matrix{Float64}(undef, 2, S)
    for i in 1:S
        c_mat[1, i] = centroids_km[i][1]
        c_mat[2, i] = centroids_km[i][2]
    end
    
    tree       = KDTree(c_mat)
    adj_thresh = sqrt(3.0) * r * 1.05

    rows_idx = Int[]
    cols_idx = Int[]
    for i in 1:S
        nbrs = inrange(tree, [centroids_km[i][1], centroids_km[i][2]], adj_thresh)
        for j in nbrs
            if j != i
                push!(rows_idx, i)
                push!(cols_idx, j)
            end
        end
    end
    
    W = sparse(rows_idx, cols_idx, ones(Float64, length(rows_idx)), S, S)
    W = max.(W, W')

    area_km2 = (3.0 * sqrt(3.0) / 2.0) * r^2

    return (
        centroids        = centroids_lonlat,
        centroids_km     = centroids_km,
        centroids_lonlat = centroids_lonlat,
        polygons         = polygons_lonlat,
        polygons_km      = polygons_km,
        polygons_lonlat  = polygons_lonlat,
        n_units          = S,
        W                = W,
        radius_km        = r,
        areas_km2        = fill(area_km2, S),
        center_lon       = center_lon,
        center_lat       = center_lat
    )
end

# ── point → unit mapping ───────────────────────────────────────────────────


"""
    map_point_to_units(
        tagging, centroids_km, center_lon, center_lat; crs=nothing, datum=WGS84Latest
    ) -> DataFrame

Assign each point observation to the nearest hexagonal unit via a KDTree on
planar km centroids. Adds `:s_idx` (1-based integer, 1 ≤ s ≤ S).

# Arguments
- `tagging`: Must contain `:lon`, `:lat` (geographic degrees).
- `centroids_km`: Vector of (x, y) km tuples from `build_hex_mesh`.
- `center_lon, center_lat`: Projection origin matching the mesh.
- `crs`: Target Coordinate Reference System (default nothing).
- `datum`: The geographic datum (default: `WGS84Latest`).
- `land_mask`: Optional boolean vector of length `S` (`true` for land units).
- `W`: Optional adjacency matrix (size `S × S`) to identify disconnected units.

# Returns
- Copy of `tagging` with `:s_idx::Int` appended.
"""
function map_point_to_units(
    tagging::DataFrame,
    centroids_km::Vector{Tuple{Float64, Float64}},
    center_lon::Real,
    center_lat::Real;
    target_col::Union{Symbol, AbstractString, Nothing} = nothing,
    crs = nothing,
    datum = WGS84Latest,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    W::Union{Nothing, AbstractMatrix} = nothing
)::DataFrame
    S = length(centroids_km)

    # Filter to active navigable units if masks or topology are provided
    active_mask = trues(S)
    if land_mask !== nothing
        active_mask .&= .!land_mask
    end
    if W !== nothing
        deg = vec(sum(W, dims=2))
        active_mask .&= (deg .> 0.0)
    end
    active_indices = findall(active_mask)
    if isempty(active_indices)
        active_indices = collect(1:S)
    end

    n_act = length(active_indices)
    c_mat = Matrix{Float64}(undef, 2, n_act)
    for (col, idx) in enumerate(active_indices)
        c_mat[1, col] = centroids_km[idx][1]
        c_mat[2, col] = centroids_km[idx][2]
    end

    # Build the KDTree for fast spatial lookups on navigable marine units
    tree = KDTree(c_mat)

    # Pre-allocate and populate the point coordinates matrix
    N = nrow(tagging)
    pts_mat = Matrix{Float64}(undef, 2, N)
    col_x, col_y = _detect_xy_columns(tagging)
    for (i, (lon, lat)) in enumerate(zip(tagging[!, col_x], tagging[!, col_y]))
        x, y = lonlat_to_xy_km(
            Float64(lon), Float64(lat);
            center_lon=center_lon, center_lat=center_lat,
            crs=crs, datum=datum
        )
        pts_mat[1, i] = x
        pts_mat[2, i] = y
    end

    # Perform nearest neighbor search for all points simultaneously
    idxs, _ = knn(tree, pts_mat, 1)

    # Append mapped units to a copy of the DataFrame
    out = copy(tagging)
    mapped_s = [active_indices[first(idx)] for idx in idxs]
    out[!, :s_idx] = mapped_s
    if !isnothing(target_col)
        out[!, Symbol(target_col)] = mapped_s
    end

    return out
end


# ── HSI loading and monthly discretization ─────────────────────────────────────

