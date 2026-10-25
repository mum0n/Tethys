# -----------------------------------------------------------------------------
# spatial_units.jl -- (GeoPartitioning) extracted verbatim from TethysBayes src/partitioning.jl
# lines 974-1724 (Section 3: Spatial Unit Assignment Dispatch and Granular Sizing).
# ------------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Section 3: High-Level Spatial Partitioning Dispatcher & Granular Sizing
# -----------------------------------------------------------------------------

"""
    _merge_undersized_polygons!(lg_polys, polys_coords, final_centroids, min_area)

Merges any polygon with area strictly less than `min_area` into its most adjacent neighbor
(sharing the longest boundary). Updates the arrays in-place.
"""
function _merge_undersized_polygons!(
    lg_polys::Vector{LibGEOS.Polygon},
    polys_coords::Vector{Vector{Tuple{Float64, Float64}}},
    final_centroids::Vector{Tuple{Float64, Float64}},
    min_area::Real
)
    if min_area <= 0.0 || length(lg_polys) <= 1
        return lg_polys, polys_coords, final_centroids
    end

    modified = true
    while modified && length(lg_polys) > 1
        modified = false
        areas = [get_polygon_area(polys_coords[i]) for i in 1:length(polys_coords)]
        under_idx = findfirst(a -> a > 0 && a < min_area, areas)

        if under_idx !== nothing
            # Find best adjacent neighbor to merge with
            p_small = lg_polys[under_idx]
            best_neighbor = 0
            best_inter_len = -1.0

            for j in 1:length(lg_polys)
                if j != under_idx
                    if LibGEOS.touches(p_small, lg_polys[j]) ||
   LibGEOS.intersects(LibGEOS.buffer(p_small, 1e-6), lg_polys[j])
                        inter = LibGEOS.intersection(LibGEOS.buffer(p_small, 1e-6), lg_polys[j])
                        inter_len = LibGEOS.area(inter)
                        if inter_len > best_inter_len
                            best_inter_len = inter_len
                            best_neighbor = j
                        end
                    end
                end
            end

            # If no touching neighbor found, find the geometrically closest one
            if best_neighbor == 0
                dists = [
                    sqrt((final_centroids[under_idx][1]-final_centroids[j][1])^2 +
                         (final_centroids[under_idx][2]-final_centroids[j][2])^2)
                    for j in 1:length(lg_polys)
                ]
                dists[under_idx] = Inf
                best_neighbor = argmin(dists)
            end

            # Merge under_idx into best_neighbor
            try
                merged_geom = LibGEOS.union(lg_polys[best_neighbor], p_small)
                if !LibGEOS.isEmpty(merged_geom)
                    if LibGEOS.geomTypeId(merged_geom) == LibGEOS.GEOS_MULTIPOLYGON
                        merged_geom = LibGEOS.convexhull(merged_geom)
                    end
                    lg_polys[best_neighbor] = merged_geom
                    polys_coords[best_neighbor] = get_coords_from_geom(merged_geom)
                    cent_g = LibGEOS.centroid(merged_geom)
                    seq = LibGEOS.getCoordSeq(cent_g)
                    final_centroids[best_neighbor] = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))

                    deleteat!(lg_polys, under_idx)
                    deleteat!(polys_coords, under_idx)
                    deleteat!(final_centroids, under_idx)
                    modified = true
                end
            catch
                # Fallback if union fails
                deleteat!(lg_polys, under_idx)
                deleteat!(polys_coords, under_idx)
                deleteat!(final_centroids, under_idx)
                modified = true
            end
        end
    end

    return lg_polys, polys_coords, final_centroids
end


"""
    _adjust_to_exact_units!(lg_polys, polys_coords, final_centroids, target_units, hull_geom)

Strictly enforces an exact number of final polygons (`target_units`).
If `K > target_units`, repeatedly merges the smallest adjacent polygon pair.
If `K < target_units`, repeatedly bisects the largest polygon along its bounding axis.
"""
function _adjust_to_exact_units!(
    lg_polys::Vector{LibGEOS.Polygon},
    polys_coords::Vector{Vector{Tuple{Float64, Float64}}},
    final_centroids::Vector{Tuple{Float64, Float64}},
    target_units::Integer,
    hull_geom
)
    if target_units <= 0 || length(lg_polys) == target_units
        return lg_polys, polys_coords, final_centroids
    end

    # Case A: Too many polygons -> iteratively merge smallest pairs
    while length(lg_polys) > target_units
        areas = [get_polygon_area(polys_coords[i]) for i in 1:length(polys_coords)]
        smallest_idx = argmin(areas)

        # Find closest/touching neighbor
        p_small = lg_polys[smallest_idx]
        best_neighbor = 0
        min_dist = Inf

        for j in 1:length(lg_polys)
            if j != smallest_idx
                d = (final_centroids[smallest_idx][1]-final_centroids[j][1])^2 +
    (final_centroids[smallest_idx][2]-final_centroids[j][2])^2
                if d < min_dist
                    min_dist = d
                    best_neighbor = j
                end
            end
        end

        if best_neighbor == 0
            break
        end

        try
            merged_geom = LibGEOS.union(lg_polys[best_neighbor], p_small)
            if !LibGEOS.isEmpty(merged_geom)
                if LibGEOS.geomTypeId(merged_geom) == LibGEOS.GEOS_MULTIPOLYGON
                    merged_geom = LibGEOS.convexhull(merged_geom)
                end
                lg_polys[best_neighbor] = merged_geom
                polys_coords[best_neighbor] = get_coords_from_geom(merged_geom)
                cent_g = LibGEOS.centroid(merged_geom)
                seq = LibGEOS.getCoordSeq(cent_g)
                final_centroids[best_neighbor] = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))

                deleteat!(lg_polys, smallest_idx)
                deleteat!(polys_coords, smallest_idx)
                deleteat!(final_centroids, smallest_idx)
            else
                deleteat!(lg_polys, smallest_idx)
                deleteat!(polys_coords, smallest_idx)
                deleteat!(final_centroids, smallest_idx)
            end
        catch
            deleteat!(lg_polys, smallest_idx)
            deleteat!(polys_coords, smallest_idx)
            deleteat!(final_centroids, smallest_idx)
        end
    end

    # Case B: Too few polygons -> iteratively bisect the largest polygon
    while length(lg_polys) < target_units
        areas = [get_polygon_area(polys_coords[i]) for i in 1:length(polys_coords)]
        largest_idx = argmax(areas)
        p_large = lg_polys[largest_idx]
        p_coords = polys_coords[largest_idx]

        if length(p_coords) < 3
            break
        end

        xs = [pt[1] for pt in p_coords if !isnan(pt[1])]
        ys = [pt[2] for pt in p_coords if !isnan(pt[2])]
        xmin, xmax = extrema(xs)
        ymin, ymax = extrema(ys)
        c_x, c_y = final_centroids[largest_idx]

        # Bisect along longer dimension
        pad = 2.0 * max(xmax - xmin, ymax - ymin, 1.0)
        box1_pts, box2_pts = if (xmax - xmin) >= (ymax - ymin)
            # Vertical split
            (
                [[(xmin - pad, ymin - pad), (c_x, ymin - pad), (c_x, ymax + pad),
                  (xmin - pad, ymax + pad), (xmin - pad, ymin - pad)]],
                [[(c_x, ymin - pad), (xmax + pad, ymin - pad), (xmax + pad, ymax + pad),
                  (c_x, ymax + pad), (c_x, ymin - pad)]]
            )
        else
            # Horizontal split
            (
                [[(xmin - pad, ymin - pad), (xmax + pad, ymin - pad), (xmax + pad, c_y),
                  (xmin - pad, c_y), (xmin - pad, ymin - pad)]],
                [[(xmin - pad, c_y), (xmax + pad, c_y), (xmax + pad, ymax + pad),
                  (xmin - pad, ymax + pad), (xmin - pad, c_y)]]
            )
        end

        try
            b1 = LibGEOS.Polygon(box1_pts)
            b2 = LibGEOS.Polygon(box2_pts)
            res1 = LibGEOS.intersection(p_large, b1)
            res2 = LibGEOS.intersection(p_large, b2)

            if !LibGEOS.isEmpty(res1) && !LibGEOS.isEmpty(res2) &&
   get_polygon_area(get_coords_from_geom(res1)) > 1e-6 &&
   get_polygon_area(get_coords_from_geom(res2)) > 1e-6
                # Replace largest with res1 and append res2
                lg_polys[largest_idx] = res1
                polys_coords[largest_idx] = get_coords_from_geom(res1)
                seq1 = LibGEOS.getCoordSeq(LibGEOS.centroid(res1))
                final_centroids[largest_idx] = (LibGEOS.getX(seq1, 1), LibGEOS.getY(seq1, 1))

                push!(lg_polys, res2)
                push!(polys_coords, get_coords_from_geom(res2))
                seq2 = LibGEOS.getCoordSeq(LibGEOS.centroid(res2))
                push!(final_centroids, (LibGEOS.getX(seq2, 1), LibGEOS.getY(seq2, 1)))
            else
                break # Cannot split further cleanly
            end
        catch
            break
        end
    end

    return lg_polys, polys_coords, final_centroids
end


"""
    assign_spatial_units(s_x, s_y; area_method=:avt, target_units=10, exact_units=false,
                         target_area=nothing, min_area=0.0, max_area=Inf, min_points=1,
                         max_points=nothing, length_scale=nothing, radius=nothing,
                         grid_resolution=nothing, aspect_ratio=1.0, prune_empty=false,
                         merge_small_polygons=false, input_polygons=nothing,
                           geom_hull=nothing, kwargs...)

Primary spatial discretization engine in `TethysBayes`. Partitions continuous spatial domains
into discrete areal units with granular control over polygon size and unit count.

### Sizing & Count Control:
- `target_units::Integer`: Desired number of final polygons (default: 10).
- `exact_units::Bool`: If `true`, dynamically enforces exactly `target_units` final polygons
  (default: false).
- `target_area::Union{Nothing, Real}`: Desired mean area per polygon in coordinate units.
  Auto-computes `target_units` and scales cell geometry.
- `min_area::Real`, `max_area::Real`: Minimum and maximum allowed polygon area.
- `merge_small_polygons::Bool`: Merges sliver/boundary-clipped polygons below `min_area`
  into neighbors.
- `prune_empty::Bool`: Removes areal units that contain 0 data observations (default: false).
- `length_scale::Real`, `radius::Real`: Explicit cell side length (`:lattice`) or hexagon
  radius (`:hexagonal`).
- `grid_resolution`: Explicit grid resolution for `:lattice` (e.g., `20` or `(rows, cols)`).
- `aspect_ratio::Real`: Rectangular cell aspect ratio for `:lattice` (`dy / dx`).

### Methods (`area_method`):
- `:avt` (default): Agglomerative Voronoi Tessellation (bottom-up merge).
- `:cvt`: Centroidal Voronoi Tessellation (Lloyd relaxation).
- `:kvt`: K-means Voronoi Tessellation (density-balanced).
- `:qvt`: Quadtree Voronoi Tessellation.
- `:bvt`: Binary Voronoi Tessellation.
- `:hvt`: Hierarchical Voronoi Tessellation.
- `:hexagonal` / `:hexbin`: Regular hexagonal cell grid.
- `:lattice`: Regular square/rectangular grid.
- `input_polygons`: User-supplied custom geometries.

Returns a `NamedTuple`:
`(centroids, polygons, adjacency_edges, graph, W, hull_coords, s_idx, s_x, s_y, s_vals,
  areas, point_counts, n_units, metrics, termination_reason)`.
"""
function assign_spatial_units(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real};
    area_method::Symbol=:avt, target_units::Union{Nothing, Integer}=nothing,
    exact_units::Bool=false, target_area::Union{Nothing, Real}=nothing,
    min_area::Real=0.0, max_area::Real=Inf, min_points::Integer=1,
    max_points::Union{Nothing, Integer}=nothing, length_scale=nothing,
    radius=nothing, grid_resolution=nothing, aspect_ratio::Real=1.0,
    prune_empty::Bool=false, merge_small_polygons::Bool=false,
    input_polygons=nothing, geom_hull=nothing, kwargs...
)
    s_coord_tuple_local = tuple.(Float64.(s_x), Float64.(s_y))
    pts_mat = hcat([[p[1], p[2]] for p in s_coord_tuple_local]...)
    n_pts = length(s_x)

    # 0. Resolve target_units and target_area
    hull_geom_raw = !isnothing(geom_hull) ? geom_hull : expand_hull(s_x, s_y, 0.0)
    domain_total_area = get_polygon_area(get_coords_from_geom(hull_geom_raw))

    eff_target_units = if !isnothing(target_units)
        Int(target_units)
    elseif !isnothing(target_area) && target_area > 0.0 && domain_total_area > 0.0
        max(1, round(Int, domain_total_area / target_area))
    else
        10
    end

    eff_max_points = isnothing(max_points) ? n_pts : Int(max_points)

    # 1. User-Supplied Custom Polygons
    if !isnothing(input_polygons)
        processed_polys = if isnothing(geom_hull)
    input_polygons
else
    [LibGEOS.intersection(p, geom_hull) for p in input_polygons]
end
        final_centroids, polys_coords, hull_coords = get_user_centroids(processed_polys)
        reason = :user_polygons
        lg_polys = LibGEOS.Polygon[p for p in processed_polys]

    # 2. Hexagonal Grid Method
    elseif area_method in [:hexagonal, :hexbin]
        hex_rad = if !isnothing(radius)
            Float64(radius)
        elseif !isnothing(target_area) && target_area > 0.0
            sqrt((2.0 * target_area) / (3.0 * sqrt(3.0)))
        elseif !isnothing(length_scale)
            Float64(length_scale)
        else
            sqrt(domain_total_area / (1.5 * sqrt(3.0) * max(1, eff_target_units)))
        end

        raw_cents, raw_polys, bbox = get_hexagonal_centroids(s_x, s_y, hex_rad)
        reason = :hexagonal_grid

        hull_geom = !isnothing(geom_hull) ? geom_hull : expand_hull(s_x, s_y, 0.1 * hex_rad)
        hull_coords = get_coords_from_geom(hull_geom)

        polys_coords = Vector{Vector{Tuple{Float64, Float64}}}()
        lg_polys = LibGEOS.Polygon[]
        final_centroids = Tuple{Float64, Float64}[]

        for (i, c) in enumerate(raw_cents)
            hex_coords = raw_polys[i]
            p_geom = LibGEOS.Polygon([[[pt[1], pt[2]] for pt in hex_coords]])
            if !isnothing(geom_hull)
                p_geom = LibGEOS.intersection(p_geom, geom_hull)
            end
            if !LibGEOS.isEmpty(p_geom) && get_polygon_area(get_coords_from_geom(p_geom)) > 1e-6
                push!(lg_polys, p_geom)
                cent_geom = LibGEOS.centroid(p_geom)
                seq = LibGEOS.getCoordSeq(cent_geom)
                push!(final_centroids, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
                push!(polys_coords, get_coords_from_geom(p_geom))
            end
        end

    # 3. Regular Lattice/Grid Partitioning
    elseif area_method == :lattice || area_method == :grid
        ls_x, ls_y = if !isnothing(grid_resolution)
            rows, cols = if grid_resolution isa Integer
                (Int(grid_resolution), Int(grid_resolution))
            else
                (Int(grid_resolution[1]), Int(grid_resolution[2]))
            end
            xmin, xmax = extrema(s_x); ymin, ymax = extrema(s_y)
            ((xmax - xmin) / max(1, cols), (ymax - ymin) / max(1, rows))
        elseif !isnothing(length_scale)
            (Float64(length_scale), Float64(length_scale) * aspect_ratio)
        elseif !isnothing(target_area) && target_area > 0.0
            side = sqrt(target_area)
            (side, side * aspect_ratio)
        else
            side = sqrt(domain_total_area / max(1, eff_target_units))
            (side, side * aspect_ratio)
        end

        xmin, xmax = extrema(s_x)
        ymin, ymax = extrema(s_y)
        x_range = collect(xmin:ls_x:xmax)
        if isempty(x_range)
            x_range = [xmin]
        end
        y_range = collect(ymin:ls_y:ymax)
        if isempty(y_range)
            y_range = [ymin]
        end
        final_centroids_raw = [(Float64(x), Float64(y)) for y in y_range, x in x_range][:]
        reason = :lattice_grid

        polys_coords = Vector{Vector{Tuple{Float64, Float64}}}()
        lg_polys = LibGEOS.Polygon[]
        final_centroids = Tuple{Float64, Float64}[]
        half_x, half_y = ls_x / 2.0, ls_y / 2.0

        for c in final_centroids_raw
            coords = [[
                [Float64(c[1]-half_x), Float64(c[2]-half_y)],
                [Float64(c[1]+half_x), Float64(c[2]-half_y)],
                [Float64(c[1]+half_x), Float64(c[2]+half_y)],
                [Float64(c[1]-half_x), Float64(c[2]+half_y)],
                [Float64(c[1]-half_x), Float64(c[2]-half_y)]
            ]]
            p_geom = LibGEOS.Polygon(coords)
            if !isnothing(geom_hull)
                p_geom = LibGEOS.intersection(p_geom, geom_hull)
            end

            if !LibGEOS.isEmpty(p_geom) && get_polygon_area(get_coords_from_geom(p_geom)) > 1e-6
                push!(lg_polys, p_geom)
                p_c = LibGEOS.centroid(p_geom)
                seq = LibGEOS.getCoordSeq(p_c)
                push!(final_centroids, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
                push!(polys_coords, get_coords_from_geom(p_geom))
            end
        end
        hull_coords = isnothing(geom_hull) ?
    [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)] :
    get_coords_from_geom(geom_hull)

    # 4. Voronoi Tessellation Methods (:avt, :cvt, :kvt, :qvt, :bvt, :hvt)
    else
        cfg = (
            target=eff_target_units,
            min_total_arealunits=Int(get(kwargs, :min_total_arealunits, 3)),
            max_total_arealunits=Int(get(kwargs, :max_total_arealunits, eff_target_units * 2)),
            min_time_slices=Int(get(kwargs, :min_time_slices, 1)),
            min_points=Int(min_points),
            max_points=eff_max_points,
            min_area=Float64(min_area),
            max_area=Float64(max_area),
            target_cv=get(kwargs, :target_cv, 1.0),
            tolerance=get(kwargs, :tolerance, 0.1),
            buffer_dist=get(kwargs, :buffer_dist, 0.5),
            t_idx=get(kwargs, :t_idx, ones(Int, length(s_x)))
        )

        hull_geom = !isnothing(geom_hull) ? geom_hull : expand_hull(s_x, s_y, cfg.buffer_dist)

        c_mid, reason = if area_method == :cvt
            get_cvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :kvt
            get_kvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :qvt
            get_qvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :bvt
            get_bvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :hvt
            get_hvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :avt
            get_avt_centroids(s_x, s_y, cfg, hull_geom)
        else
            error("Unknown partitioning method: $area_method. " *
      "Choose from :avt, :cvt, :kvt, :qvt, :bvt, :hvt, :hexagonal, :lattice.")
        end

        polys_coords, v_edges = get_voronoi_polygons_and_edges(c_mid, hull_geom)
        final_centroids = Tuple{Float64, Float64}[]
        lg_polys = LibGEOS.Polygon[]
        for p_coords in polys_coords
            if isempty(p_coords)
                continue
            end
            if p_coords[1] != p_coords[end]
                push!(p_coords, p_coords[1])
            end
            lg_p = LibGEOS.Polygon([[ [pt[1], pt[2]] for pt in p_coords ]])
            push!(lg_polys, lg_p)
            cent_g = LibGEOS.centroid(lg_p)
            seq = LibGEOS.getCoordSeq(cent_g)
            push!(final_centroids, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
        end
        hull_coords = get_coords_from_geom(hull_geom)
    end

    # -------------------------------------------------------------------------
    # Granular Post-Processing: Merging, Pruning, and Exact Units Enforcement
    # -------------------------------------------------------------------------

    is_regular_grid = area_method in [:hexagonal, :hexbin, :lattice, :grid]

    # A. Merge small polygons below min_area (irregular Voronoi tessellations only)
    # Regular grids (hexagons/lattices) are uniform geometry; merging cells creates
    # irregular multi-polygon aggregations and destroys regular grid properties.
    if !is_regular_grid && (merge_small_polygons || (min_area > 0.0))
        _merge_undersized_polygons!(lg_polys, polys_coords, final_centroids, min_area)
    end

    # B. Strict exact units enforcement
    if exact_units && eff_target_units > 0
        if is_regular_grid
            # For regular grids, rank cells by observation density and proximity
            # to data center without merging or deforming hexagonal geometries.
            if length(final_centroids) > eff_target_units
                cand_mat = hcat([[c[1], c[2]] for c in final_centroids]...)
                cand_tree = KDTree(cand_mat)
                cand_assigns, _ = knn(cand_tree, pts_mat, 1)
                pt_counts = zeros(Int, length(final_centroids))
                for a in cand_assigns
                    pt_counts[a[1]] += 1
                end

                mean_x = mean(s_x)
                mean_y = mean(s_y)
                dists_to_center = [
                    (c[1] - mean_x)^2 + (c[2] - mean_y)^2 for c in final_centroids
                ]
                ranked_indices = sort(
                    1:length(final_centroids),
                    by = i -> (-pt_counts[i], dists_to_center[i])
                )
                keep_indices = sort(ranked_indices[1:eff_target_units])

                lg_polys = lg_polys[keep_indices]
                polys_coords = polys_coords[keep_indices]
                final_centroids = final_centroids[keep_indices]
            end
        else
            hull_for_adjust = !isnothing(geom_hull) ? geom_hull : expand_hull(s_x, s_y, 0.1)
            _adjust_to_exact_units!(
                lg_polys, polys_coords, final_centroids, eff_target_units, hull_for_adjust
            )
        end
    end

    # C. Point Assignments
    c_mat = hcat([[c[1], c[2]] for c in final_centroids]...)
    tree = KDTree(c_mat)
    assigns_knn, _ = knn(tree, pts_mat, 1)
    new_assigns = [a[1] for a in assigns_knn]

    # D. Prune empty units if requested
    if prune_empty && length(final_centroids) > 1
        active_units = unique(new_assigns)
        if length(active_units) < length(final_centroids)
            lg_polys = lg_polys[active_units]
            polys_coords = polys_coords[active_units]
            final_centroids = final_centroids[active_units]

            # Reassign points
            c_mat = hcat([[c[1], c[2]] for c in final_centroids]...)
            tree = KDTree(c_mat)
            assigns_knn, _ = knn(tree, pts_mat, 1)
            new_assigns = [a[1] for a in assigns_knn]
        end
    end

    # E. Construct Final Adjacency Graph W
    n_units = length(final_centroids)
    g = SimpleGraph(n_units)
    for i in 1:n_units, j in (i+1):n_units
        if LibGEOS.touches(lg_polys[i], lg_polys[j]) ||
   LibGEOS.intersects(LibGEOS.buffer(lg_polys[i], 1e-7), lg_polys[j])
            add_edge!(g, i, j)
        end
    end
    g = ensure_connected!(g, final_centroids)
    W = Float64.(Graphs.adjacency_matrix(g))

    v_edges = Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}[]
    for e in Graphs.edges(g)
        push!(v_edges, (final_centroids[src(e)], final_centroids[dst(e)]))
    end

    # F. Compute Detailed Area and Point Count Metrics
    areas = [get_polygon_area(p) for p in polys_coords]
    point_counts = [count(==(i), new_assigns) for i in 1:n_units]
    metrics = (
        mean_density = mean(point_counts),
        sd_density = length(point_counts) > 1 ? std(point_counts) : 0.0,
        cv_density = length(point_counts) > 1 ?
        std(point_counts) / (mean(point_counts) + 1e-9) : 0.0,
        min_density = isempty(point_counts) ? 0 : minimum(point_counts),
        max_density = isempty(point_counts) ? 0 : maximum(point_counts),
        mean_area = mean(areas),
        sd_area = length(areas) > 1 ? std(areas) : 0.0,
        cv_area = length(areas) > 1 ? std(areas) / (mean(areas) + 1e-9) : 0.0,
        min_area = isempty(areas) ? 0.0 : minimum(areas),
        max_area = isempty(areas) ? 0.0 : maximum(areas),
        total_area = sum(areas)
    )

    wkt_val = if haskey(kwargs, :wkt) && !isnothing(kwargs[:wkt])
        string(kwargs[:wkt])
    elseif haskey(kwargs, :crs) && !isnothing(kwargs[:crs])
        string(kwargs[:crs])
    elseif haskey(kwargs, :proj) && !isnothing(kwargs[:proj])
        string(kwargs[:proj])
    else
        nothing
    end

    return (
        centroids = final_centroids,
        polygons = polys_coords,
        adjacency_edges = v_edges,
        graph = g,
        W = W,
        hull_coords = hull_coords,
        s_idx = new_assigns,
        assignments = new_assigns,
        s_x = s_x,
        s_y = s_y,
        s_vals = collect(1:size(W, 1)),
        areas = areas,
        point_counts = point_counts,
        n_units = n_units,
        metrics = metrics,
        termination_reason = reason,
        wkt = wkt_val
    )
end

# Overloaded signatures for Tuples & DataFrames
assign_spatial_units(coords::Vector{<:Tuple{Real, Real}}; kwargs...) =
    assign_spatial_units([p[1] for p in coords], [p[2] for p in coords]; kwargs...)

function assign_spatial_units(
    df::DataFrame;
    x::Union{Symbol, AbstractString, Nothing}=nothing,
    y::Union{Symbol, AbstractString, Nothing}=nothing,
    target_col::Union{Symbol, AbstractString, Nothing}=nothing,
    kwargs...
)
    col_x, col_y = _detect_xy_columns(df; x=x, y=y)
    au = assign_spatial_units(df[!, col_x], df[!, col_y]; kwargs...)
    if !isnothing(target_col)
        df[!, Symbol(target_col)] = au.assignments
    end
    return au
end


"""
    assign_spatial_units_inferred(adjacency_matrix; iterations=50, learning_rate=0.1,
      buffer_dist=0.5, input_polygons=nothing, wkt=nothing)

Infers spatial node positions and creates Voronoi boundaries when only a neighborhood
adjacency matrix `W` is available (e.g. Scottish Lip Cancer dataset). Uses a force-directed
spring layout to determine relative spatial coordinates.
"""
function assign_spatial_units_inferred(
    adjacency_matrix; iterations=50, learning_rate=0.1, buffer_dist=0.5, input_polygons=nothing,
    wkt=nothing, crs=nothing, proj=nothing
)
    nAU = size(adjacency_matrix, 1)

    wkt_val = if !isnothing(wkt)
        string(wkt)
    elseif !isnothing(crs)
        string(crs)
    elseif !isnothing(proj)
        string(proj)
    else
        nothing
    end

    if input_polygons !== nothing && !isempty(input_polygons)
        final_centroids_geoms = [LibGEOS.centroid(p) for p in input_polygons]
        final_centroids = map(final_centroids_geoms) do g_pt
            seq = LibGEOS.getCoordSeq(g_pt)
            (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
        end

        collection = LibGEOS.GeometryCollection(LibGEOS.Polygon[p for p in input_polygons])
        united_geom = LibGEOS.unaryUnion(collection)
        hull_coords_output = get_coords_from_geom(united_geom)

        adjacency_edges_output = Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}[]
        for i in 1:nAU
            g1 = input_polygons[i]
            for j in (i+1):nAU
                g2 = input_polygons[j]
                if LibGEOS.touches(g1, g2)
                    push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                else
                    g1_buffered = LibGEOS.buffer(g1, 1e-6)
                    if LibGEOS.intersects(g1_buffered, g2)
                        push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                    end
                end
            end
        end

        polys_output = [get_coords_from_geom(p) for p in input_polygons]
        g_final = SimpleGraph(nAU)
        for i in 1:nAU, j in (i+1):nAU
            if adjacency_matrix[i, j] != 0
                add_edge!(g_final, i, j)
            end
        end
        g_final = ensure_connected!(g_final, final_centroids)
    else
        g_initial = SimpleGraph(adjacency_matrix)
        side = ceil(Int, sqrt(nAU))
        initial_centroids = [(Float64(i % side), Float64(i ÷ side)) for i in 0:(nAU-1)]
        centroids_vec = [SVector{2, Float64}(c) for c in initial_centroids]

        for _ in 1:iterations
            new_centroids_vec = copy(centroids_vec)
            for i in 1:nAU
                neighbors_i = Graphs.neighbors(g_initial, i)
                if !isempty(neighbors_i)
                    avg_pos = sum(centroids_vec[n] for n in neighbors_i) / length(neighbors_i)
                    new_centroids_vec[i] = centroids_vec[i] +
            learning_rate * (avg_pos - centroids_vec[i])
                end
            end
            centroids_vec = new_centroids_vec
        end

        forced_centroids = [(p[1], p[2]) for p in centroids_vec]
        fx = [c[1] for c in forced_centroids]
        fy = [c[2] for c in forced_centroids]
        hull_geom = expand_hull(fx, fy, buffer_dist)
        hull_coords_output = get_coords_from_geom(hull_geom)

        polys_coords_raw, _ = get_voronoi_polygons_and_edges(forced_centroids, hull_geom)
        final_centroids = Vector{Tuple{Float64, Float64}}(undef, length(polys_coords_raw))
        polys_output = polys_coords_raw

        for (idx, poly_coord_list) in enumerate(polys_coords_raw)
            if length(poly_coord_list) >= 3
                if poly_coord_list[1] != poly_coord_list[end]
                    push!(poly_coord_list, poly_coord_list[1])
                end
                lg_poly = LibGEOS.Polygon([[ [p[1], p[2]] for p in poly_coord_list ]])
                seq = LibGEOS.getCoordSeq(LibGEOS.centroid(lg_poly))
                final_centroids[idx] = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
            else
                final_centroids[idx] = forced_centroids[idx]
            end
        end

        g_final = SimpleGraph(nAU)
        for i in 1:nAU, j in (i+1):nAU
            if adjacency_matrix[i, j] != 0
                add_edge!(g_final, i, j)
            end
        end
        g_final = ensure_connected!(g_final, final_centroids)
        adjacency_edges_output = Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}[]
        for e in Graphs.edges(g_final)
            push!(adjacency_edges_output, (final_centroids[src(e)], final_centroids[dst(e)]))
        end
    end

    return (
        centroids = final_centroids,
        polygons = polys_output,
        adjacency_edges = adjacency_edges_output,
        graph = g_final,
        W = Float64.(adjacency_matrix),
        hull_coords = hull_coords_output,
        s_idx = collect(1:nAU),
        s_x = [c[1] for c in final_centroids[1:nAU]],
        s_y = [c[2] for c in final_centroids[1:nAU]],
        s_vals = collect(1:nAU),
        termination_reason = "positions inferred from adjacency matrix",
        wkt = wkt_val
    )
end


