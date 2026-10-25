# -----------------------------------------------------------------------------
# geometry.jl -- (GeoPartitioning) extracted verbatim from TethysBayes src/partitioning.jl
# lines 1-282 (Section 1: Geometric Primitives and Boundary Utilities).
# ------------------------------------------------------------------------------
"""
    partitioning.jl

Spatial and spatiotemporal domain decomposition, Voronoi/hexagonal tessellation,
areal unit assignment, graph connectivity, and spatial block cross-validation for TethysBayes.

Version: v1.0.0
"""

# -----------------------------------------------------------------------------
# Section 1: Geometric Primitives & Boundary Utilities
# -----------------------------------------------------------------------------

"""
    expand_hull(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, buffer_dist::Real)

Computes the convex hull of 2D points `(s_x, s_y)` and expands it by `buffer_dist`
using `LibGEOS.buffer`. Returns a `LibGEOS` polygon geometry.
"""
function expand_hull(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, buffer_dist::Real)
    if isempty(s_x) || isempty(s_y)
        return LibGEOS.Polygon([[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]])
    end
    coords_vec = [[Float64(s_x[i]), Float64(s_y[i])] for i in 1:length(s_x)]
    points_geom = LibGEOS.MultiPoint(coords_vec)
    hull = LibGEOS.convexhull(points_geom)
    if buffer_dist > 0.0
        return LibGEOS.buffer(hull, Float64(buffer_dist))
    else
        return hull
    end
end

expand_hull(coords::Vector{<:Tuple{Real, Real}}, buffer_dist::Real) =
    expand_hull([p[1] for p in coords], [p[2] for p in coords], buffer_dist)


"""
    get_polygon_area(poly_coords::AbstractVector)

Calculates the 2D surface area of a polygon using the Shoelace formula.
Coordinates are expected as a vector of `(x, y)` tuples or vectors.
"""
function get_polygon_area(poly_coords::AbstractVector)
    valid_pts = Tuple{Float64, Float64}[]
    for p in poly_coords
        x, y = float(p[1]), float(p[2])
        if !isnan(x) && !isinf(x) && !isnan(y) && !isinf(y)
            push!(valid_pts, (x, y))
        end
    end

    if length(valid_pts) > 1 && valid_pts[1] == valid_pts[end]
        valid_pts = valid_pts[1:end-1]
    end

    if length(valid_pts) < 3
        return 0.0
    end

    x = [p[1] for p in valid_pts]
    y = [p[2] for p in valid_pts]

    # Shoelace formula
    return 0.5 * abs(dot(x, circshift(y, 1)) - dot(y, circshift(x, 1)))
end

get_polygon_area(s_x::AbstractVector, s_y::AbstractVector) = get_polygon_area(tuple.(s_x, s_y))


"""
    is_valid_polygon_coords(poly_coords)

Validates whether a collection of vertices contains at least 3 non-NaN/non-Inf points.
"""
function is_valid_polygon_coords(poly_coords)
    valid_pts = [
        p for p in poly_coords
        if !isnan(p[1]) && !isinf(p[1]) && !isnan(p[2]) && !isinf(p[2])
    ]
    return length(valid_pts) >= 3
end


"""
    get_coords_from_geom(geom)

Extracts coordinate tuples `Vector{Tuple{Float64, Float64}}` from `LibGEOS` geometries
(`Point`, `Polygon`, `MultiPolygon`, `LineString`, `LinearRing`).
"""
function get_coords_from_geom(geom)
    coords = Tuple{Float64, Float64}[]
    if geom === nothing
        return coords
    end
    local type_id = -1
    try
        type_id = LibGEOS.geomTypeId(geom)
        if type_id == LibGEOS.GEOS_POINT
            seq = LibGEOS.getCoordSeq(geom)
            push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
            return coords
        elseif type_id == LibGEOS.GEOS_POLYGON
            ring = LibGEOS.exteriorRing(geom)
            n = LibGEOS.numPoints(ring)
            for i in 1:n
                p = LibGEOS.getPoint(ring, i)
                seq = LibGEOS.getCoordSeq(p)
                push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
            end
        elseif type_id == LibGEOS.GEOS_MULTIPOLYGON
            n_geoms = LibGEOS.numGeometries(geom)
            for i in 1:n_geoms
                poly = LibGEOS.getGeometryN(geom, i)
                ring = LibGEOS.exteriorRing(poly)
                n = LibGEOS.numPoints(ring)
                for j in 1:n
                    p = LibGEOS.getPoint(ring, j)
                    seq = LibGEOS.getCoordSeq(p)
                    push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
                end
                if i < n_geoms
                    push!(coords, (NaN, NaN))
                end
            end
        elseif type_id in [LibGEOS.GEOS_LINESTRING, LibGEOS.GEOS_LINEARRING]
            n = LibGEOS.numPoints(geom)
            for i in 1:n
                p = LibGEOS.getPoint(geom, i)
                seq = LibGEOS.getCoordSeq(p)
                push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
            end
        end
    catch e
        @warn "Coordinate extraction failed for LibGEOS type $type_id: $e"
    end
    return coords
end


"""
    get_kde_seeds(coords, target_u)

Fast density-weighted seeding for spatial tessellation. Uses KDTree to estimate local
point densities in O(N log N) without allocating large pairwise distance matrices.
"""
function get_kde_seeds(coords::Vector{<:Tuple{Real, Real}}, target_u::Integer)
    u_pts = unique(coords)
    n = length(u_pts)
    if n == 0
        return Tuple{Float64, Float64}[]
    elseif n <= target_u
        return [(Float64(p[1]), Float64(p[2])) for p in u_pts]
    end

    # Fast KD-Tree 5-nearest-neighbors density proxy
    pts_mat = hcat([[Float64(p[1]), Float64(p[2])] for p in u_pts]...)
    tree = KDTree(pts_mat)
    k_eval = min(6, n)
    _, dists = knn(tree, pts_mat, k_eval)

    # Inverse mean distance as density weight
    weights = [1.0 / (mean(d[2:end]) + 1e-6) for d in dists]
    idx = StatsBase.sample(1:n, Weights(weights), min(target_u, n), replace=false)
    return [(Float64(u_pts[i][1]), Float64(u_pts[i][2])) for i in idx]
end


"""
    get_voronoi_polygons_and_edges(centroids, hull_geom, tol=1e-7)

Generates Voronoi polygons for `centroids`, clips them to `hull_geom` via `LibGEOS`,
and extracts contiguous neighbor edges. Handles edge cases for 0, 1, and 2 centroids.
"""
function get_voronoi_polygons_and_edges(
    centroids::Vector{<:Tuple{Real, Real}}, hull_geom, tol::Real=1e-7
)
    n_c = length(centroids)
    if n_c == 0
        return Tuple{Float64, Float64}[][],
               Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}[]
    elseif n_c == 1
        return [get_coords_from_geom(hull_geom)],
               Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}[]
    elseif n_c == 2
        p1 = (Float64(centroids[1][1]), Float64(centroids[1][2]))
        p2 = (Float64(centroids[2][1]), Float64(centroids[2][2]))
        mid = ((p1[1] + p2[1]) / 2.0, (p1[2] + p2[2]) / 2.0)
        dx, dy = p2[1] - p1[1], p2[2] - p1[2]
        px, py = -dy, dx

        env = LibGEOS.envelope(hull_geom)
        hull_bbox_coords = get_coords_from_geom(env)
        L = if !isempty(hull_bbox_coords)
            min_x, max_x = extrema(c[1] for c in hull_bbox_coords)
            min_y, max_y = extrema(c[2] for c in hull_bbox_coords)
            2.0 * sqrt((max_x - min_x)^2 + (max_y - min_y)^2) + 1.0
        else
            1e7
        end

        pt1 = (mid[1] + L*px, mid[2] + L*py)
        pt2 = (mid[1] - L*px, mid[2] - L*py)
        side1_pts = [pt1, pt2, (pt2[1] - L*dx, pt2[2] - L*dy), (pt1[1] - L*dx, pt1[2] - L*dy), pt1]
        poly1_box = LibGEOS.Polygon([[[p[1], p[2]] for p in side1_pts]])
        side2_pts = [pt1, pt2, (pt2[1] + L*dx, pt2[2] + L*dy), (pt1[1] + L*dx, pt1[2] + L*dy), pt1]
        poly2_box = LibGEOS.Polygon([[[p[1], p[2]] for p in side2_pts]])
        res1 = LibGEOS.intersection(hull_geom, poly1_box)
        res2 = LibGEOS.intersection(hull_geom, poly2_box)
        return [get_coords_from_geom(res1), get_coords_from_geom(res2)], [(p1, p2)]
    end

    # Deduplicate centroids
    u_centroids = unique(centroids)
    if length(u_centroids) < n_c
        u_polys, u_edges = get_voronoi_polygons_and_edges(u_centroids, hull_geom, tol)
        return [u_polys[findfirst(==(c), u_centroids)] for c in centroids], u_edges
    end

    # 3+ centroids: Delaunay -> Voronoi
    pts_dt = [(Float64(c[1]), Float64(c[2])) for c in centroids]
    tri = triangulate(pts_dt)
    hull_coords = get_coords_from_geom(hull_geom)
    xs = [p[1] for p in hull_coords if !isnan(p[1])]
    ys = [p[2] for p in hull_coords if !isnan(p[2])]
    if isempty(xs) || isempty(ys)
        return [Tuple{Float64, Float64}[] for _ in 1:length(centroids)], []
    end
    
    bbox = (minimum(xs), maximum(xs), minimum(ys), maximum(ys))
    vorn = voronoi(tri)
    final_coords = [Tuple{Float64, Float64}[] for _ in 1:length(centroids)]
    valid_geoms = Dict{Int, Any}()

    for i in each_generator(vorn)
        if i < 1 || i > length(centroids)
            continue
        end
        vertices = get_polygon_coordinates(vorn, i, bbox)
        if !isempty(vertices)
            poly_pts = [[v[1], v[2]] for v in vertices]
            if poly_pts[1] != poly_pts[end]
                push!(poly_pts, poly_pts[1])
            end
            try
                lg_poly = LibGEOS.Polygon([poly_pts])
                clipped = LibGEOS.intersection(lg_poly, hull_geom)
                if !LibGEOS.isEmpty(clipped) &&
   LibGEOS.geomTypeId(clipped) in [LibGEOS.GEOS_POLYGON, LibGEOS.GEOS_MULTIPOLYGON]
                    final_coords[i] = get_coords_from_geom(clipped)
                    valid_geoms[i] = clipped
                end
            catch
            end
        end
    end

    v_edges = Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}[]
    active_ids = sort(collect(keys(valid_geoms)))
    for idx in 1:length(active_ids)
        i = active_ids[idx]
        g1 = valid_geoms[i]
        for jdx in idx+1:length(active_ids)
            j = active_ids[jdx]
            g2 = valid_geoms[j]
            if LibGEOS.touches(g1, g2)
                push!(v_edges, ((Float64(centroids[i][1]), Float64(centroids[i][2])),
                                (Float64(centroids[j][1]), Float64(centroids[j][2]))))
            else
                g1_b = LibGEOS.buffer(g1, Float64(tol))
                if LibGEOS.intersects(g1_b, g2)
                    push!(v_edges, ((Float64(centroids[i][1]), Float64(centroids[i][2])),
                                    (Float64(centroids[j][1]), Float64(centroids[j][2]))))
                end
            end
        end
    end

    return final_coords, v_edges
end

# -----------------------------------------------------------------------------
# Section 2: Polygon Decomposition, Centroid Extraction & Continuous Surface Reconstruction
# -----------------------------------------------------------------------------

"""
    poly_xy(poly)

Converts polygon representations (Vector of (x,y) tuples, vectors, or arrays)
into two vectors of `Float64` coordinates `(xs, ys)`, filtering out `NaN` values.

# Inputs
- `poly`: Collection of 2D vertex coordinate pairs `(x, y)`.

# Outputs
- `Tuple{Vector{Float64}, Vector{Float64}}`: Coordinate vectors `(xs, ys)`.
"""
function poly_xy(poly)
    xs = Float64[]
    ys = Float64[]
    for p in poly
        x = float(p[1])
        y = float(p[2])
        if !isnan(x) && !isnan(y)
            push!(xs, x)
            push!(ys, y)
        end
    end
    return xs, ys
end

const _poly_xy = poly_xy

"""
    centroids_xy(centroids)

Safely extracts x and y coordinates from a collection of centroids into two
`Float64` vectors.

# Inputs
- `centroids`: Collection of coordinate tuples or 2D vectors.

# Outputs
- `Tuple{Vector{Float64}, Vector{Float64}}`: Coordinate vectors `(xs, ys)`.
"""
function centroids_xy(centroids)
    xs = Float64[]
    ys = Float64[]
    for c in centroids
        push!(xs, float(c[1]))
        push!(ys, float(c[2]))
    end
    return xs, ys
end

const _centroids_xy = centroids_xy

"""
    extract_polygons(au)

Robustly extracts polygon vertex arrays from a collection, NamedTuple
(`:polygons_lonlat`, `:polygons`, or `:polygons_km`), or Vector.

# Inputs
- `au`: Spatial areal units representation (mesh, NamedTuple, or AbstractDict).

# Outputs
- `Vector`: Array of polygon vertex coordinates.
"""
function extract_polygons(au)
    if au isa AbstractVector
        return au
    elseif au isa NamedTuple
        if hasproperty(au, :polygons_lonlat) && !isnothing(au.polygons_lonlat)
            return au.polygons_lonlat
        elseif hasproperty(au, :polygons) && !isnothing(au.polygons)
            return au.polygons
        elseif hasproperty(au, :polygons_km) && !isnothing(au.polygons_km)
            return au.polygons_km
        end
    elseif au isa AbstractDict
        if haskey(au, :polygons_lonlat) && !isnothing(au[:polygons_lonlat])
            return au[:polygons_lonlat]
        elseif haskey(au, :polygons) && !isnothing(au[:polygons])
            return au[:polygons]
        elseif haskey(au, :polygons_km) && !isnothing(au[:polygons_km])
            return au[:polygons_km]
        end
    end
    throw(ArgumentError(
        "Could not extract polygons. Expected vector or NamedTuple with :polygons."
    ))
end

const _extract_polygons = extract_polygons

"""
    extract_centroids(au)

Robustly extracts centroid coordinates from a collection, NamedTuple
(`:centroids_lonlat`, `:centroids`, or `:centroids_km`), or Vector.

# Inputs
- `au`: Spatial areal units representation (mesh, NamedTuple, or AbstractDict).

# Outputs
- `Vector`: Array of centroid coordinate tuples.
"""
function extract_centroids(au)
    if au isa AbstractVector
        return au
    elseif au isa NamedTuple
        if hasproperty(au, :centroids_lonlat) && !isnothing(au.centroids_lonlat)
            return au.centroids_lonlat
        elseif hasproperty(au, :centroids) && !isnothing(au.centroids)
            return au.centroids
        elseif hasproperty(au, :centroids_km) && !isnothing(au.centroids_km)
            return au.centroids_km
        end
    elseif au isa AbstractDict
        if haskey(au, :centroids_lonlat) && !isnothing(au[:centroids_lonlat])
            return au[:centroids_lonlat]
        elseif haskey(au, :centroids) && !isnothing(au[:centroids])
            return au[:centroids]
        elseif haskey(au, :centroids_km) && !isnothing(au[:centroids_km])
            return au[:centroids_km]
        end
    end
    throw(ArgumentError(
        "Could not extract centroids. Expected vector or NamedTuple with :centroids."
    ))
end

const _extract_centroids = extract_centroids

"""
    build_spatial_domain_grid(coords; grid_res=150, ext_factor=0.05, domain_bbox=nothing)
    -> Tuple{Vector{Float64}, Vector{Float64}}

Constructs regular 1D coordinate vectors `(grid_x, grid_y)` spanning the continuous
spatial domain bounding box with proportional boundary extension padding.

# Arguments
- `coords::AbstractMatrix{<:Real}`: Observation coordinate matrix (N × 2).
- `grid_res::Int`: Number of grid points along each coordinate axis (default 150).
- `ext_factor::Real`: Domain expansion fraction applied to bounding box (default 0.05).
- `domain_bbox`: Optional explicit `(x_min, x_max, y_min, y_max)` bounding box.

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: Uniform grid coordinate vectors `(grid_x, grid_y)`.
"""
function build_spatial_domain_grid(
    coords::AbstractMatrix{<:Real};
    grid_res::Int = 150,
    ext_factor::Real = 0.05,
    domain_bbox = nothing
)
    if size(coords, 2) < 2
        throw(ArgumentError("Coordinates must have at least 2 columns for 2D spatial domain grid."))
    end

    x_min, x_max = !isnothing(domain_bbox) ?
        (domain_bbox[1], domain_bbox[2]) : extrema(coords[:, 1])
    y_min, y_max = !isnothing(domain_bbox) ?
        (domain_bbox[3], domain_bbox[4]) : extrema(coords[:, 2])

    dx = max(x_max - x_min, 1e-6)
    dy = max(y_max - y_min, 1e-6)

    x_start = x_min - ext_factor * dx
    x_end   = x_max + ext_factor * dx
    y_start = y_min - ext_factor * dy
    y_end   = y_max + ext_factor * dy

    grid_x = collect(range(x_start, stop = x_end, length = grid_res))
    grid_y = collect(range(y_start, stop = y_end, length = grid_res))

    return grid_x, grid_y
end

const _build_spatial_domain_grid = build_spatial_domain_grid

"""
    reconstruct_spatial_surface(coords, values; grid_res=150, ext_factor=0.05,
                                length_scale=nothing, nugget=1e-4, domain_bbox=nothing)
    -> NamedTuple

Reconstructs a smooth continuous random surface over a regular 2D grid spanning the
spatial domain using regularized Matérn-3/2 kernel interpolation.

# Mathematical Formulation
Given observation coordinates \$\\mathbf{x}_i \\in \\mathbb{R}^2\$ and scalar effect
realizations \$z_i\$, the continuous spatial field is represented as a Gaussian process:
```math
z(\\mathbf{x}_0) = \\bar{z} + \\sum_{i=1}^N \\alpha_i \\cdot k(\\|\\mathbf{x}_0 - \\mathbf{x}_i\\|)
```
where the kernel is the Matérn-3/2 covariance function:
```math
k(r) = \\left(1 + \\frac{\\sqrt{3}r}{\\ell}\\right) \\exp\\left(-\\frac{\\sqrt{3}r}{\\ell}\\right)
```
with correlation length scale \$\\ell\$ automatically estimated from domain scale,
and regularized dual coefficients \$\\boldsymbol{\\alpha} = (\\mathbf{K} + \\lambda \\mathbf{I})^{-1}(\\mathbf{z} - \\bar{z})\$.

# Inputs
- `coords::AbstractMatrix{<:Real}`: 2D coordinate matrix of observations (N × 2).
- `values::AbstractVector{<:Real}`: Scalar values at each observation location (length N).
- `grid_res::Int`: Lattice resolution along each axis (default 150).
- `ext_factor::Real`: Domain expansion fraction for boundary padding (default 0.05).
- `length_scale`: Correlation length scale \$\\ell\$ (default auto-estimated from spatial scale).
- `nugget::Real`: Tikhonov regularization factor (default \$10^{-4}\$).
- `domain_bbox`: Optional explicit bounding box `(x_min, x_max, y_min, y_max)`.

# Outputs
- `NamedTuple`: `(grid_x = Vector, grid_y = Vector, surface = Matrix)`.
"""
function reconstruct_spatial_surface(
    coords::AbstractMatrix{<:Real},
    values::AbstractVector{<:Real};
    grid_res::Int = 150,
    ext_factor::Real = 0.05,
    length_scale::Union{Real, Nothing} = nothing,
    nugget::Real = 1e-4,
    domain_bbox = nothing
)
    c_mat = Matrix{Float64}(coords)
    v_vec = vec(collect(Float64, values))

    if size(c_mat, 1) != length(v_vec)
        throw(DimensionMismatch(
            "Coordinates rows ($(size(c_mat, 1))) and values length ($(length(v_vec))) must match."
        ))
    end

    valid_mask = .!isnan.(v_vec) .& .!isnan.(c_mat[:, 1]) .& .!isnan.(c_mat[:, 2])
    c_clean = c_mat[valid_mask, :]
    v_clean = v_vec[valid_mask]

    n_pts = length(v_clean)
    if n_pts == 0
        error("No valid non-NaN coordinate and value pairs available for surface reconstruction.")
    end

    grid_x, grid_y = build_spatial_domain_grid(
        c_clean; grid_res = grid_res, ext_factor = ext_factor, domain_bbox = domain_bbox
    )

    if n_pts == 1 || all(isapprox(v_clean[1]), v_clean)
        surface_const = fill(v_clean[1], length(grid_x), length(grid_y))
        return (grid_x = grid_x, grid_y = grid_y, surface = surface_const)
    end

    x_min, x_max = extrema(c_clean[:, 1])
    y_min, y_max = extrema(c_clean[:, 2])
    dx = max(x_max - x_min, 1e-6)
    dy = max(y_max - y_min, 1e-6)

    ell = if !isnothing(length_scale) && length_scale > 0
        Float64(length_scale)
    else
        max(1.5 * sqrt(dx * dy / (pi * max(n_pts, 1))), 0.05 * max(dx, dy))
    end

    v_mean = mean(v_clean)
    v_cent = v_clean .- v_mean
    v_var = var(v_clean)
    reg_val = max(nugget * v_var, 1e-8)

    sq3 = sqrt(3.0)
    @inline function _matern32(d_val::Float64, scale::Float64)
        r = d_val / scale
        return (1.0 + sq3 * r) * exp(-sq3 * r)
    end

    surface_z = zeros(Float64, length(grid_x), length(grid_y))

    if n_pts <= 2000
        K = Matrix{Float64}(undef, n_pts, n_pts)
        for j in 1:n_pts
            xj, yj = c_clean[j, 1], c_clean[j, 2]
            for i in 1:j
                d_ij = hypot(c_clean[i, 1] - xj, c_clean[i, 2] - yj)
                k_val = _matern32(d_ij, ell)
                K[i, j] = k_val
                K[j, i] = k_val
            end
            K[j, j] += reg_val
        end

        alpha_weights = cholesky(Symmetric(K)) \ v_cent

        for (j, gy) in enumerate(grid_y)
            for (i, gx) in enumerate(grid_x)
                s_val = 0.0
                for k in 1:n_pts
                    d_k = hypot(gx - c_clean[k, 1], gy - c_clean[k, 2])
                    s_val += alpha_weights[k] * _matern32(d_k, ell)
                end
                surface_z[i, j] = v_mean + s_val
            end
        end
    else
        for (j, gy) in enumerate(grid_y)
            for (i, gx) in enumerate(grid_x)
                weight_sum = 0.0
                val_sum = 0.0
                for k in 1:n_pts
                    d_k = hypot(gx - c_clean[k, 1], gy - c_clean[k, 2])
                    w_k = _matern32(d_k, ell)
                    weight_sum += w_k
                    val_sum += w_k * v_clean[k]
                end
                surface_z[i, j] = weight_sum > 1e-12 ? (val_sum / weight_sum) : v_mean
            end
        end
    end

    return (grid_x = grid_x, grid_y = grid_y, surface = surface_z)
end

const _reconstruct_spatial_surface = reconstruct_spatial_surface


