# -----------------------------------------------------------------------------
# tessellation.jl -- (GeoPartitioning) extracted verbatim from TethysBayes src/partitioning.jl
# lines 283-973 (Section 2: Spatial Tessellation and Centroid Algorithms).
# ------------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Section 2: Spatial Tessellation & Centroid Algorithms
# -----------------------------------------------------------------------------

"""
    get_cvt_centroids(s_x, s_y, cfg, hull_geom)

Centroidal Voronoi Tessellation (CVT) via Lloyd's relaxation algorithm.
Iteratively relaxes centroids to the geometric center of their Voronoi polygons.
"""
function get_cvt_centroids(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom
)
    coords = tuple.(Float64.(s_x), Float64.(s_y))
    if length(coords) <= cfg.min_total_arealunits
        return [(mean(s_x), mean(s_y))], "not_enough_points_to_tessellate"
    end

    u_pts = unique(coords)
    idx = StatsBase.sample(1:length(u_pts), min(cfg.target, length(u_pts)), replace=false)
    curr_centroids = [u_pts[i] for i in idx]
    termination_reason = "max_iterations"
    last_mean_density, last_cv = 0.0, 0.0

    pts_mat = hcat([[p[1], p[2]] for p in coords]...)

    for iter in 1:100
        polys, _ = get_voronoi_polygons_and_edges(curr_centroids, hull_geom)
        new_centroids = Tuple{Float64, Float64}[]
        shifts = Float64[]

        for i in 1:length(polys)
            poly_coords = polys[i]
            area = get_polygon_area(poly_coords)

            if length(poly_coords) > 2 && area >= cfg.min_area && area <= cfg.max_area
                lg_poly = LibGEOS.Polygon([[[p[1], p[2]] for p in poly_coords]])
                cent_geom = LibGEOS.centroid(lg_poly)
                seq = LibGEOS.getCoordSeq(cent_geom)
                new_c = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
                push!(shifts, sqrt(sum((new_c .- curr_centroids[i]).^2)))
                push!(new_centroids, new_c)
            else
                push!(new_centroids, curr_centroids[i])
            end
        end

        if isempty(shifts) || mean(shifts) < cfg.tolerance
            termination_reason = "convergence"
            break
        end

        # Fast KDTree assignment
        c_mat = hcat([[c[1], c[2]] for c in new_centroids]...)
        tree = KDTree(c_mat)
        assigns, _ = knn(tree, pts_mat, 1)
        assign_indices = [a[1] for a in assigns]
        counts = [count(==(i), assign_indices) for i in 1:length(new_centroids)]

        if isempty(counts)
            termination_reason = "no_units_formed"
            break
        end

        curr_mean_density = mean(counts)
        if abs(curr_mean_density - last_mean_density) < cfg.tolerance && iter > 1
            termination_reason = "density_convergence"
            break
        end
        last_mean_density = curr_mean_density

        cv_val = std(counts) / (mean(counts) + 1e-9)
        if abs(cv_val - last_cv) < cfg.tolerance && iter > 1
            termination_reason = "cv_convergence"
            break
        end
        last_cv = cv_val

        if mean(counts) < cfg.min_points
            termination_reason = "min_points_violation"
            break
        end

        curr_centroids = new_centroids
    end

    return curr_centroids, termination_reason
end


"""
    get_kvt_centroids(s_x, s_y, cfg, hull_geom)

K-means Voronoi Tessellation (KVT) moves centroids toward the center of mass
of data points inside each cell, balancing point density.
"""
function get_kvt_centroids(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom
)
    coords = tuple.(Float64.(s_x), Float64.(s_y))
    if length(coords) <= cfg.min_total_arealunits
        return [(mean(s_x), mean(s_y))], "not_enough_points_to_tessellate"
    end

    u_pts = unique(coords)
    idx_init = StatsBase.sample(1:length(u_pts), min(cfg.target, length(u_pts)), replace=false)
    c_iter = [u_pts[i] for i in idx_init]
    data = tuple.(coords, cfg.t_idx)
    pts_mat = hcat([[p[1], p[2]] for p in coords]...)

    damping = 0.7
    termination_reason = "max_iterations"
    last_mean_density, last_cv = 0.0, 0.0

    for iter in 1:100
        old_centroids = copy(c_iter)
        c_mat = hcat([[c[1], c[2]] for c in c_iter]...)
        tree = KDTree(c_mat)
        assigns_knn, _ = knn(tree, pts_mat, 1)
        assigns = [a[1] for a in assigns_knn]

        polys_coords, _ = get_voronoi_polygons_and_edges(c_iter, hull_geom)

        for k in 1:length(c_iter)
            idx_cluster = findall(==(k), assigns)
            ts_count = length(unique([data[j][2] for j in idx_cluster]))
            area = k <= length(polys_coords) ? get_polygon_area(polys_coords[k]) : 0.0
            area_ok = (area > 0) && area >= cfg.min_area && area <= cfg.max_area

            if !isempty(idx_cluster) && length(idx_cluster) >= cfg.min_points &&
   ts_count >= cfg.min_time_slices && area_ok
                mean_x = mean(data[j][1][1] for j in idx_cluster)
                mean_y = mean(data[j][1][2] for j in idx_cluster)
                c_iter[k] = ((1.0 - damping) * old_centroids[k][1] + damping * mean_x,
                             (1.0 - damping) * old_centroids[k][2] + damping * mean_y)
            end
        end

        counts = [count(==(k), assigns) for k in 1:length(c_iter)]
        if isempty(counts)
            termination_reason = "no_units_formed"
            break
        end

        curr_mean_density = mean(counts)
        if abs(curr_mean_density - last_mean_density) < cfg.tolerance && iter > 1
            termination_reason = "density_convergence"
            break
        end
        last_mean_density = curr_mean_density

        cv_val = std(counts) / (mean(counts) + 1e-9)
        if abs(cv_val - last_cv) < cfg.tolerance && iter > 1
            termination_reason = "cv_convergence"
            break
        end
        last_cv = cv_val

        if mean(counts) < cfg.min_points
            termination_reason = "min_points_violation"
            break
        end

        damping *= 0.99
    end

    return c_iter, termination_reason
end


"""
    get_qvt_centroids(s_x, s_y, cfg, hull_geom)

Quadtree Voronoi Tessellation (QVT) recursively subdivides regions into quadrants
based on local coordinate medians, adapting to high-density clusters.
"""
function get_qvt_centroids(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom
)
    coords = tuple.(Float64.(s_x), Float64.(s_y))
    if length(coords) <= cfg.min_total_arealunits
        return [(mean(s_x), mean(s_y))], "not_enough_points_to_tessellate"
    end

    data = tuple.(coords, cfg.t_idx)
    regions = [data]
    unsplittable = Set{UInt64}()
    effective_min_p = max(1, cfg.min_points)

    if length(data) < 2 * effective_min_p
        return [(mean(p[1][1] for p in data), mean(p[1][2] for p in data))],
       "initial_data_too_small_to_tessellate"
    end

    termination_reason = "max_units_reached"
    last_mean_density, last_cv = 0.0, 0.0
    cnt = 0

    while length(regions) < cfg.max_total_arealunits
        cnt += 1
        counts = length.(regions)
        curr_mean_density = mean(counts)
        cv_val = std(counts) / (curr_mean_density + 1e-9)

        if cnt > 3
            if last_mean_density > 0.0 &&
   (abs(curr_mean_density - last_mean_density) < cfg.tolerance ||
    abs(cv_val - last_cv) < cfg.tolerance)
                if length(regions) >= cfg.target && all(c -> c <= cfg.max_points, counts)
                    termination_reason = "converged_constraints_satisfied"
                    break
                elseif abs(cv_val - cfg.target_cv) < cfg.tolerance
                    termination_reason = "converged_target_cv"
                    break
                end
            end
        end

        last_mean_density = curr_mean_density
        last_cv = cv_val

        viable_indices = findall(
    r -> length(r) >= max(2, effective_min_p) && objectid(r) ∉ unsplittable, regions
)
        if isempty(viable_indices)
            break
        end

        target_idx = viable_indices[argmax([length(regions[i]) for i in viable_indices])]
        target_region = regions[target_idx]
        xs_r = [p[1][1] for p in target_region]
        ys_r = [p[1][2] for p in target_region]

        if length(unique(xs_r)) > 1 || length(unique(ys_r)) > 1
            mx = length(unique(xs_r)) > 1 ? median(xs_r) : xs_r[1]
            my = length(unique(ys_r)) > 1 ? median(ys_r) : ys_r[1]
            r_splits = [
                filter(p -> p[1][1] <= mx && p[1][2] <= my, target_region),
                filter(p -> p[1][1] > mx && p[1][2] <= my, target_region),
                filter(p -> p[1][1] <= mx && p[1][2] > my, target_region),
                filter(p -> p[1][1] > mx && p[1][2] > my, target_region)
            ]
        else
            mid = length(target_region) ÷ 2
            r_splits = [target_region[1:mid], target_region[mid+1:end], [], []]
        end

        valid_splits = filter(r -> length(r) >= effective_min_p, r_splits)
        if length(valid_splits) < 2
            push!(unsplittable, objectid(target_region))
            continue
        end

        deleteat!(regions, target_idx)
        append!(regions, valid_splits)
    end

    final_centroids = [(mean(p[1][1] for p in r), mean(p[1][2] for p in r)) for r in regions]
    return final_centroids, termination_reason
end


"""
    get_bvt_centroids(s_x, s_y, cfg, hull_geom)

Binary Voronoi Tessellation (BVT) recursively bisects regions along their principal axis of
  maximum variance.
"""
function get_bvt_centroids(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom
)
    coords = tuple.(Float64.(s_x), Float64.(s_y))
    if length(coords) <= cfg.min_total_arealunits
        return [(mean(s_x), mean(s_y))], "not_enough_points_to_tessellate"
    end

    data = tuple.(coords, cfg.t_idx)
    regions = [data]
    effective_min_p = max(1, cfg.min_points)

    if length(data) < 2 * effective_min_p
        return [(mean(p[1][1] for p in data), mean(p[1][2] for p in data))],
       "initial_data_too_small_to_tessellate"
    end

    unsplittable = Set{UInt64}()
    termination_reason = "max_units_reached"
    last_mean_density, last_cv = 0.0, 0.0
    cnt = 0

    while length(regions) < cfg.max_total_arealunits
        cnt += 1
        counts = length.(regions)
        curr_mean_density = mean(counts)
        cv_val = std(counts) / (curr_mean_density + 1e-9)

        if cnt > 3
            if last_mean_density > 0.0 &&
   (abs(curr_mean_density - last_mean_density) < cfg.tolerance ||
    abs(cv_val - last_cv) < cfg.tolerance)
                if length(regions) >= cfg.target && all(c -> c <= cfg.max_points, counts)
                    termination_reason = "converged_constraints_satisfied"
                    break
                elseif (abs(cv_val - cfg.target_cv) < cfg.tolerance)
                    termination_reason = "converged_target_cv"
                    break
                end
            end
        end

        last_mean_density = curr_mean_density
        last_cv = cv_val

        viable_indices = findall(
    r -> length(r) >= max(2, effective_min_p) && objectid(r) ∉ unsplittable, regions
)
        if cnt > 3 && isempty(viable_indices)
            termination_reason = "cannot_split_further"
            break
        end

        violators = filter(i -> length(regions[i]) > cfg.max_points, viable_indices)
        must_split = length(regions) < cfg.min_total_arealunits
        want_split = length(regions) < cfg.target || !isempty(violators)
        candidates = if must_split || want_split
    isempty(violators) ? viable_indices : violators
else
    []
end

        if isempty(candidates)
            termination_reason = "constraints_satisfied"; break
        end

        target_idx = candidates[argmax([length(regions[i]) for i in candidates])]
        target = regions[target_idx]

        xs = [p[1][1] for p in target]; ys = [p[1][2] for p in target]
        var_x = length(xs) > 1 ? var(xs) : 0.0
        var_y = length(ys) > 1 ? var(ys) : 0.0
        dim = var_x > var_y ? 1 : 2

        if var_x > 1e-9 || var_y > 1e-9
            vals = [p[1][dim] for p in target]
            med = length(unique(vals)) > 1 ? median(vals) : vals[1]
            r1 = filter(p -> p[1][dim] <= med, target)
            r2 = filter(p -> p[1][dim] > med, target)
        else
            mid = length(target) ÷ 2
            r1, r2 = target[1:mid], target[mid+1:end]
        end

        v1 = length(r1) >= effective_min_p &&
     length(unique([p[2] for p in r1])) >= cfg.min_time_slices
        v2 = length(r2) >= effective_min_p &&
     length(unique([p[2] for p in r2])) >= cfg.min_time_slices

        if !v1 || !v2
             push!(unsplittable, objectid(target))
             continue
        end

        tentative_regions = copy(regions)
        deleteat!(tentative_regions, target_idx)
        push!(tentative_regions, r1, r2)

        candidate_centroids = [
    (mean(p[1][1] for p in r), mean(p[1][2] for p in r))
    for r in tentative_regions
]
        polys_coords, _ = get_voronoi_polygons_and_edges(candidate_centroids, hull_geom)

        area_violation = any(
            p_coords -> !is_valid_polygon_coords(p_coords) ||
            get_polygon_area(p_coords) < cfg.min_area,
            polys_coords
        )

        if area_violation && length(tentative_regions) > cfg.min_total_arealunits
             push!(unsplittable, objectid(target))
             continue
        end

        regions = tentative_regions
    end

    final_centroids_candidate = [
    (mean(p[1][1] for p in r), mean(p[1][2] for p in r))
    for r in regions
]

    if length(final_centroids_candidate) < cfg.min_total_arealunits
        all_pts_x = [p[1][1] for p in data]
        all_pts_y = [p[1][2] for p in data]
        return [(mean(all_pts_x), mean(all_pts_y))], "insufficient_units_error"
    else
        return final_centroids_candidate, termination_reason
    end
end


"""
    get_hvt_centroids(s_x, s_y, cfg, hull_geom; max_iter=500)

Hierarchical Voronoi Tessellation (HVT) combines initial k-means seeding
with adaptive splitting of overcrowded partitions.
"""
function get_hvt_centroids(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom; max_iter=500
)
    coords = tuple.(Float64.(s_x), Float64.(s_y))
    dist(p1, p2) = sqrt((p1[1]-p2[1])^2 + (p1[2]-p2[2])^2)

    pts_matrix = hcat([[p[1], p[2]] for p in coords]...)
    k_target = max(1, cfg.min_total_arealunits)

    R = Clustering.kmeans(pts_matrix, k_target)
    curr_centroids = [(R.centers[1, i], R.centers[2, i]) for i in 1:size(R.centers, 2)]

    last_mean_density, last_cv = 0.0, 0.0
    status = "max_iterations_reached"

    for i in 1:max_iter
        c_mat = hcat([[c[1], c[2]] for c in curr_centroids]...)
        tree = KDTree(c_mat)
        assigns_knn, _ = knn(tree, pts_matrix, 1)
        s_idx = [a[1] for a in assigns_knn]
        counts = [count(==(k), s_idx) for k in 1:length(curr_centroids)]

        curr_mean_density = mean(counts)
        cv_val = std(counts) / (curr_mean_density + 1e-9)

        if i > 5
            if abs(curr_mean_density - last_mean_density) < cfg.tolerance ||
   abs(cv_val - last_cv) < cfg.tolerance
                if length(curr_centroids) >= cfg.target && all(c -> c <= cfg.max_points, counts)
                    status = "converged_constraints_satisfied"
                    break
                elseif abs(cv_val - cfg.target_cv) < cfg.tolerance
                    status = "converged_target_cv"
                    break
                end
            end
        end

        last_mean_density = curr_mean_density
        last_cv = cv_val

        # Refinement step (Lloyd's update)
        new_centroids = Tuple{Float64, Float64}[]
        for k in 1:length(curr_centroids)
            group_pts = coords[s_idx .== k]
            if !isempty(group_pts)
                push!(new_centroids, (
    mean(p[1] for p in group_pts), mean(p[2] for p in group_pts)
))
            else
                push!(new_centroids, curr_centroids[k])
            end
        end

        if all(dist(new_centroids[j], curr_centroids[j]) < cfg.tolerance
       for j in 1:length(curr_centroids))
             if length(curr_centroids) < cfg.max_total_arealunits &&
   (length(curr_centroids) < cfg.target || any(counts .> cfg.max_points))
                 idx_to_split = argmax(counts)
                 group_pts = coords[s_idx .== idx_to_split]
                 if length(group_pts) >= 2 * cfg.min_points
                    new_seeds = [
                        (mean(p[1] for p in group_pts) * 0.99,
                         mean(p[2] for p in group_pts) * 0.99),
                        (mean(p[1] for p in group_pts) * 1.01,
                         mean(p[2] for p in group_pts) * 1.01)
                    ]
                    deleteat!(curr_centroids, idx_to_split)
                    append!(curr_centroids, new_seeds)
                    continue
                 end
             end
             status = "converged_stable_positions"
             break
        end

        curr_centroids = new_centroids
    end

    return curr_centroids, status
end


"""
    get_avt_centroids(s_x, s_y, cfg, hull_geom)

Agglomerative Voronoi Tessellation (AVT) begins with fine-grained partitions and merges
underpopulated or starved units bottom-up until area and point count constraints are satisfied.
"""
function get_avt_centroids(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom
)
    coords = tuple.(Float64.(s_x), Float64.(s_y))
    if length(coords) <= cfg.min_total_arealunits
        return [(mean(p[1] for p in coords), mean(p[2] for p in coords))],
       "not_enough_points_to_tessellate"
    end

    u_pts = unique(coords)
    c_init = get_kde_seeds(u_pts, min(length(u_pts), cfg.max_total_arealunits))
    data = tuple.(coords, cfg.t_idx)
    curr_c = [SVector{2, Float64}(c) for c in c_init]

    termination_reason = "min_units_reached"
    last_mean_density, last_cv = 0.0, 0.0

    while length(curr_c) > cfg.min_total_arealunits
        assigns = [Int[] for _ in 1:length(curr_c)]
        for i in 1:length(data)
            d_pt = data[i][1]
            dist_idx = argmin([sum((d_pt .- c).^2) for c in curr_c])
            push!(assigns[dist_idx], i)
        end
        counts = length.(assigns)

        polys_coords, _ = get_voronoi_polygons_and_edges([Tuple(c) for c in curr_c], hull_geom)
        areas = fill(0.0, length(curr_c))
        for i in 1:min(length(curr_c), length(polys_coords))
            areas[i] = get_polygon_area(polys_coords[i])
        end

        violators = Int[]
        for k in 1:length(curr_c)
            ts_count = length(unique([data[idx][2] for idx in assigns[k]]))
            is_invalid_count = counts[k] < cfg.min_points
            is_invalid_time = ts_count < cfg.min_time_slices
            is_invalid_area = (areas[k] > 0 && areas[k] < cfg.min_area) ||
                  (areas[k] > cfg.max_area)
            if is_invalid_count || is_invalid_time || is_invalid_area
                push!(violators, k)
            end
        end

        curr_mean_density = mean(counts)
        cv_val = std(counts) / (mean(counts) + 1e-9)
        if last_mean_density > 0.0 &&
   (abs(curr_mean_density - last_mean_density) < cfg.tolerance ||
    abs(cv_val - last_cv) < cfg.tolerance)
            termination_reason = "tolerance_reached"
            break
        end
        last_mean_density = curr_mean_density
        last_cv = cv_val

        candidates_indices = isempty(violators) ? collect(1:length(curr_c)) : violators
        v_counts = [counts[k] for k in candidates_indices]
        target_idx = candidates_indices[argmin(v_counts)]

        dists = [sum((curr_c[target_idx] .- curr_c[j]).^2) for j in 1:length(curr_c)]
        dists[target_idx] = Inf
        neighbor_idx = argmin(dists)

        total_n = counts[target_idx] + counts[neighbor_idx]
        curr_c[neighbor_idx] = (curr_c[target_idx] .* counts[target_idx] .+
                        curr_c[neighbor_idx] .* counts[neighbor_idx]) ./ (total_n + 1e-9)
        deleteat!(curr_c, target_idx)
    end

    return [Tuple(c) for c in curr_c], termination_reason
end


"""
    get_lattice_centroids(s_x, s_y, length_scale)

Generates regular square lattice centroids covering the bounding box of `(s_x, s_y)`.
"""
function get_lattice_centroids(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, length_scale::Real
)
    if isempty(s_x) || isempty(s_y)
        return Tuple{Float64, Float64}[], 0, 0, (0.0, 0.0, 0.0, 0.0)
    end

    xmin, xmax = extrema(s_x)
    ymin, ymax = extrema(s_y)

    x_range = collect(xmin:length_scale:xmax)
    y_range = collect(ymin:length_scale:ymax)

    if isempty(x_range)
        x_range = [xmin]
    end
    if isempty(y_range)
        y_range = [ymin]
    end

    rows = length(y_range)
    cols = length(x_range)
    centroids = [(Float64(x), Float64(y)) for y in y_range, x in x_range][:]

    return centroids, rows, cols, (xmin, xmax, ymin, ymax)
end


"""
    get_hexagonal_centroids(s_x, s_y, radius)

Generates regular hexagonal grid centroids and polygons covering the bounding box of `(s_x, s_y)`.
Hexagonal packing provides uniform distance to all 6 neighbors, eliminating axial orientation bias.
"""
function get_hexagonal_centroids(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, radius::Real
)
    if isempty(s_x) || isempty(s_y)
        return Tuple{Float64, Float64}[], Vector{Tuple{Float64, Float64}}[], (0.0, 0.0, 0.0, 0.0)
    end

    xmin, xmax = extrema(s_x)
    ymin, ymax = extrema(s_y)

    dx = sqrt(3.0) * radius
    dy = 1.5 * radius

    centroids = Tuple{Float64, Float64}[]
    polygons = Vector{Vector{Tuple{Float64, Float64}}}()

    row = 0
    y_curr = ymin
    while y_curr <= ymax + radius
        x_offset = isodd(row) ? (dx / 2.0) : 0.0
        x_curr = xmin - radius + x_offset
        while x_curr <= xmax + radius
            push!(centroids, (x_curr, y_curr))
            # 6 vertices of regular hexagon
            hex_verts = Tuple{Float64, Float64}[]
            for k in 0:5
                angle = deg2rad(30.0 + 60.0 * k)
                vx = x_curr + radius * cos(angle)
                vy = y_curr + radius * sin(angle)
                push!(hex_verts, (vx, vy))
            end
            push!(hex_verts, hex_verts[1]) # closed polygon
            push!(polygons, hex_verts)
            x_curr += dx
        end
        y_curr += dy
        row += 1
    end

    return centroids, polygons, (xmin, xmax, ymin, ymax)
end


"""
    get_user_centroids(input_polygons)

Extracts centroids, closed polygon coordinates, and combined bounding hull from
  user-provided `LibGEOS` polygons.
"""
function get_user_centroids(input_polygons)
    geoms = LibGEOS.Polygon[p for p in input_polygons]
    n = length(geoms)
    centroids = Vector{Tuple{Float64, Float64}}(undef, n)
    polys_coords = Vector{Vector{Tuple{Float64, Float64}}}(undef, n)

    for i in 1:n
        poly = geoms[i]
        cent_geom = LibGEOS.centroid(poly)
        seq = LibGEOS.getCoordSeq(cent_geom)
        centroids[i] = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
        polys_coords[i] = get_coords_from_geom(poly)
    end

    collection = LibGEOS.GeometryCollection(geoms)
    united = LibGEOS.unaryUnion(collection)
    hull_coords = get_coords_from_geom(united)

    return centroids, polys_coords, hull_coords
end


"""
    load_shapefile_to_libgeos(filepath::String)

Loads a shapefile (.shp) and converts its geometries into `LibGEOS` objects.
"""
function load_shapefile_to_libgeos(filepath::String)
    table = Shapefile.Table(filepath)
    geoms = [LibGEOS.read_geom(row.geometry) for row in table]
    return geoms, table
end


"""
    compute_network_transfer_matrix(au_src::NamedTuple, au_dest::NamedTuple;
                                    method::Symbol = :area_weighted, kwargs...)

Computes the sparse linear transfer operator matrix `P` of dimension
`N_dest × N_src` that projects spatial areal fields from a source partition
`au_src` to a destination partition `au_dest`.

### Methods
- `:area_weighted` (default): Computes polygon overlap fraction
  `P_{j,i} = Area(B_j ∩ A_i) / Area(B_j)` using `LibGEOS.intersection`.
- `:idw` / `:centroid_idw`: Inverse Distance Weighting using KD-tree nearest neighbors.
"""
function compute_network_transfer_matrix(
    au_src::NamedTuple,
    au_dest::NamedTuple;
    method::Symbol = :area_weighted,
    kwargs...
)
    _extract_cents(au) = if hasproperty(au, :centroids) && !isnothing(au.centroids)
        au.centroids
    elseif hasproperty(au, :centroids_lonlat) && !isnothing(au.centroids_lonlat)
        au.centroids_lonlat
    elseif hasproperty(au, :centroids_km) && !isnothing(au.centroids_km)
        au.centroids_km
    else
        error("Areal unit partition has no centroids field.")
    end

    src_cents = _extract_cents(au_src)
    dest_cents = _extract_cents(au_dest)
    n_src = length(src_cents)
    n_dest = length(dest_cents)

    if n_src == n_dest && src_cents == dest_cents
        return spdiagm(0 => ones(Float64, n_dest))
    end

    # Determine coordinate scale (geographic degrees vs projected planar meters/km)
    is_geo_coords(pts) = !isempty(pts) && all(abs(c[1]) <= 180.5 && abs(c[2]) <= 90.5 for c in pts)
    src_is_geo = is_geo_coords(src_cents)

    resolve_au_geom(au, ref_geo) = begin
        if ref_geo && hasproperty(au, :polygons_lonlat) && !isempty(au.polygons_lonlat)
            cents = hasproperty(au, :centroids_lonlat) ? au.centroids_lonlat : _extract_cents(au)
            return (polygons = au.polygons_lonlat, centroids = cents)
        elseif !ref_geo && hasproperty(au, :polygons_km) && !isempty(au.polygons_km)
            cents = hasproperty(au, :centroids_km) ? au.centroids_km : _extract_cents(au)
            return (polygons = au.polygons_km, centroids = cents)
        elseif hasproperty(au, :polygons) && !isempty(au.polygons)
            cents = _extract_cents(au)
            return (polygons = au.polygons, centroids = cents)
        else
            return (polygons = Vector{Vector{Tuple{Float64, Float64}}}(), centroids = _extract_cents(au))
        end
    end

    geom_src = resolve_au_geom(au_src, src_is_geo)
    geom_dest = resolve_au_geom(au_dest, src_is_geo)

    has_polys = !isempty(geom_src.polygons) && !isempty(geom_dest.polygons) &&
                length(geom_src.polygons) == n_src && length(geom_dest.polygons) == n_dest

    if has_polys && method != :centroid_idw && method != :idw
        try
            function _to_libgeos_valid_polygon(poly_coords)
                if isempty(poly_coords) || length(poly_coords) < 3
                    return nothing
                end
                pts = [[Float64(pt[1]), Float64(pt[2])] for pt in poly_coords if !isnan(pt[1]) && !isnan(pt[2])]
                length(pts) < 3 && return nothing
                pts[1] != pts[end] && push!(pts, pts[1])
                length(pts) < 4 && return nothing
                try
                    lg_p = LibGEOS.Polygon([pts])
                    if !LibGEOS.isValid(lg_p)
                        lg_p = LibGEOS.buffer(lg_p, 0.0)
                    end
                    return lg_p
                catch
                    return nothing
                end
            end

            src_lg_polys = [_to_libgeos_valid_polygon(p) for p in geom_src.polygons]
            dest_lg_polys = [_to_libgeos_valid_polygon(p) for p in geom_dest.polygons]

            src_bboxes = Vector{Tuple{Float64, Float64, Float64, Float64}}(undef, n_src)
            for i in 1:n_src
                p = geom_src.polygons[i]
                xs = [pt[1] for pt in p if !isnan(pt[1])]
                ys = [pt[2] for pt in p if !isnan(pt[2])]
                src_bboxes[i] = (isempty(xs) || isempty(ys)) ?
                    (0.0, 0.0, 0.0, 0.0) :
                    (minimum(xs), maximum(xs), minimum(ys), maximum(ys))
            end

            P = spzeros(Float64, n_dest, n_src)
            all_valid = true

            for j in 1:n_dest
                dest_p = dest_lg_polys[j]
                if dest_p === nothing
                    all_valid = false
                    break
                end
                dest_area = get_polygon_area(geom_dest.polygons[j])
                dest_area <= 1e-9 && continue

                dp = geom_dest.polygons[j]
                d_xs = [pt[1] for pt in dp if !isnan(pt[1])]
                d_ys = [pt[2] for pt in dp if !isnan(pt[2])]
                (isempty(d_xs) || isempty(d_ys)) && continue
                d_min_x, d_max_x = minimum(d_xs), maximum(d_xs)
                d_min_y, d_max_y = minimum(d_ys), maximum(d_ys)

                total_overlap = 0.0
                for i in 1:n_src
                    src_p = src_lg_polys[i]
                    src_p === nothing && continue

                    s_min_x, s_max_x, s_min_y, s_max_y = src_bboxes[i]
                    if d_min_x > s_max_x || d_max_x < s_min_x ||
                       d_min_y > s_max_y || d_max_y < s_min_y
                        continue
                    end

                    if LibGEOS.intersects(dest_p, src_p)
                        inter_geom = LibGEOS.intersection(dest_p, src_p)
                        if !LibGEOS.isEmpty(inter_geom)
                            inter_area = get_polygon_area(get_coords_from_geom(inter_geom))
                            if inter_area > 1e-9
                                w = inter_area / dest_area
                                P[j, i] = w
                                total_overlap += w
                            end
                        end
                    end
                end

                if total_overlap > 1e-6
                    P[j, :] ./= total_overlap
                else
                    c_j = geom_dest.centroids[j]
                    dists = [hypot(c_j[1] - geom_src.centroids[i][1], c_j[2] - geom_src.centroids[i][2]) for i in 1:n_src]
                    nearest_i = argmin(dists)
                    P[j, nearest_i] = 1.0
                end
            end

            if all_valid && all(sum(P, dims=2) .> 0.99)
                return P
            end
        catch
            # Fall back to centroid IDW
        end
    end

    # Centroid Inverse Distance Weighting Fallback via KDTree
    src_coords = hcat([[c[1], c[2]] for c in src_cents]...)
    dest_coords = hcat([[c[1], c[2]] for c in dest_cents]...)

    kdtree = KDTree(src_coords)
    k_nn = min(4, n_src)
    idxs, dists = knn(kdtree, dest_coords, k_nn, true)

    P = spzeros(Float64, n_dest, n_src)
    for j in 1:n_dest
        cur_idxs = idxs[j]
        cur_dists = dists[j]

        if cur_dists[1] <= 1e-6
            P[j, cur_idxs[1]] = 1.0
        else
            weights = 1.0 ./ (cur_dists .^ 2)
            weights ./= sum(weights)
            for (idx, w) in zip(cur_idxs, weights)
                P[j, idx] = w
            end
        end
    end

    return P
end


