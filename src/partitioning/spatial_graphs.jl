# -----------------------------------------------------------------------------
# spatial_graphs.jl -- (GeoPartitioning) extracted verbatim from TethysBayes src/partitioning.jl
# lines 1725-2057 (Section 4: Spatial Graph Topology and Weighting Matrices).
# ------------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Section 4: Spatial Graph Topology & Weighting Matrix Utilities
# -----------------------------------------------------------------------------

"""
    spatial_weights_matrix(W::AbstractMatrix; style::Symbol=:binary)

Transforms an adjacency matrix `W` into standard spatial econometric / statistical weight schemes:
- `:binary` (default): Unweighted 0/1 adjacency.
- `:row_standardized` / `:row_norm`: Row-normalized matrix where rows sum to 1 (`W[i,:] /
  sum(W[i,:])`).
- `:variance_stabilized`: Normalized by `sqrt(row_sum)`.
"""
function spatial_weights_matrix(W::AbstractMatrix; style::Symbol=:binary)
    if style == :binary
        return Float64.(W)
    elseif style in [:row_standardized, :row_norm]
        row_sums = vec(sum(W, dims=2))
        inv_sums = [s > 0 ? 1.0 / s : 0.0 for s in row_sums]
        return Diagonal(inv_sums) * Float64.(W)
    elseif style == :variance_stabilized
        row_sums = vec(sum(W, dims=2))
        inv_sqrt = [s > 0 ? 1.0 / sqrt(s) : 0.0 for s in row_sums]
        return Diagonal(inv_sqrt) * Float64.(W)
    else
        error("Unknown spatial weighting style: $style. " *
      "Choose from :binary, :row_standardized, :variance_stabilized.")
    end
end

"""
    spatial_weights_matrix(
        coords::Vector{<:Tuple{Real, Real}};
        method::Symbol = :knn,
        k::Integer = 4,
        radius::Union{Nothing, Real} = nothing,
        style::Symbol = :binary
    )

Construct a spatial weights matrix directly from coordinate tuples using k-nearest
neighbors or distance thresholding, applying the specified spatial weighting style.
"""
function spatial_weights_matrix(
    coords::Vector{<:Tuple{Real, Real}};
    method::Symbol = :knn,
    k::Integer = 4,
    radius::Union{Nothing, Real} = nothing,
    style::Symbol = :binary
)
    g, W = if method == :knn
        spatial_knn_graph(coords, k)
    elseif method == :radius
        rad = radius === nothing ?
            error("radius parameter required when method=:radius") : radius
        spatial_radius_graph(coords, rad)
    else
        error("Unknown spatial graph method: $(method). Supported: :knn, :radius.")
    end
    return spatial_weights_matrix(W; style = style)
end


"""
    spatial_knn_graph(coords, k::Integer)

Constructs a k-nearest neighbor spatial graph and binary adjacency matrix directly from
  coordinate points.
"""
function spatial_knn_graph(coords::Vector{<:Tuple{Real, Real}}, k::Integer)
    n = length(coords)
    k_adj = min(k, n - 1)
    pts_mat = hcat([[Float64(p[1]), Float64(p[2])] for p in coords]...)
    tree = KDTree(pts_mat)
    idxs, _ = knn(tree, pts_mat, k_adj + 1)

    g = SimpleGraph(n)
    for i in 1:n
        for neighbor in idxs[i][2:end]
            add_edge!(g, i, neighbor)
        end
    end
    W = Float64.(Graphs.adjacency_matrix(g))
    return g, W
end


"""
    spatial_radius_graph(coords, radius::Real)

Constructs a distance-threshold spatial graph connecting all points within distance `radius`.
"""
function spatial_radius_graph(coords::Vector{<:Tuple{Real, Real}}, radius::Real)
    n = length(coords)
    pts_mat = hcat([[Float64(p[1]), Float64(p[2])] for p in coords]...)
    tree = KDTree(pts_mat)
    idxs = inrange(tree, pts_mat, Float64(radius))

    g = SimpleGraph(n)
    for i in 1:n
        for neighbor in idxs[i]
            if neighbor != i
                add_edge!(g, i, neighbor)
            end
        end
    end
    W = Float64.(Graphs.adjacency_matrix(g))
    return g, W
end


"""
    ensure_connected!(g::SimpleGraph, centroids::Vector{<:Tuple{Real, Real}})

Ensures a spatial graph `g` is fully connected by identifying disjoint components
and adding minimal bridging edges between their closest centroid pairs.
"""
function ensure_connected!(g::SimpleGraph, centroids::Vector{<:Tuple{Real, Real}})
    comps = connected_components(g)
    if length(comps) <= 1
        return g
    end

    n_nodes = length(centroids)
    c_mat = hcat([[Float64(c[1]), Float64(c[2])] for c in centroids]...)
    tree = KDTree(c_mat)

    # 1. Add minimal bridging edge from each component to nearest external node
    for comp in comps
        comp_set = Set(comp)
        best_d = Inf
        best_u = 0
        best_v = 0
        k_search = min(n_nodes, max(4, ceil(Int, sqrt(n_nodes))))
        for u in comp
            idxs, dists = knn(tree, [centroids[u][1], centroids[u][2]], k_search)
            for (k_idx, v) in enumerate(idxs)
                if !(v in comp_set)
                    if dists[k_idx] < best_d
                        best_d = dists[k_idx]
                        best_u = u
                        best_v = v
                        break
                    end
                end
            end
        end
        if best_u != 0 && best_v != 0 && !has_edge(g, best_u, best_v)
            add_edge!(g, best_u, best_v)
        end
    end

    # 2. Iteratively bridge any remaining disjoint component centroids
    comps = connected_components(g)
    while length(comps) > 1
        n_c = length(comps)
        comp_centroids = [
            mean([[Float64(centroids[n][1]), Float64(centroids[n][2])] for n in c])
            for c in comps
        ]
        tree_c = KDTree(hcat(comp_centroids...))
        idxs, _ = knn(tree_c, comp_centroids[1], min(2, n_c))
        target_idx = idxs[min(2, length(idxs))]
        u = comps[1][1]
        v = comps[target_idx][1]
        add_edge!(g, u, v)
        comps = connected_components(g)
    end

    return g
end




"""
    scaling_factor_bym2(W::AbstractMatrix)

Computes the BYM2 structured GMRF scaling factor from adjacency matrix `W` by
constructing the singular ICAR precision matrix `Q = Diagonal(row_sums) - W`
and evaluating the geometric mean of non-zero eigenvalues (Riebler et al., 2016).
"""
function scaling_factor_bym2(W::AbstractMatrix)
    if size(W, 1) != size(W, 2)
        error("Adjacency matrix W must be square.")
    end
    Q = Diagonal(vec(sum(W, dims=2))) - W
    return _compute_scaling_factor(Q)
end

function _compute_scaling_factor(Q_template)
    eigenvalues = eigvals(Symmetric(Matrix(Q_template)))
    non_zero_eigenvalues = eigenvalues[eigenvalues .> 1e-12]
    if isempty(non_zero_eigenvalues)
        return 1.0
    end
    return exp(-mean(log.(non_zero_eigenvalues)))
end


"""
    adjacency_matrix_to_nb(W)

Converts a binary adjacency matrix `W` into a neighbor-list representation `Vector{Vector{Int}}`.
"""
function adjacency_matrix_to_nb(W::AbstractMatrix)
    nau = size(W, 1)
    nb = [Int[] for _ in 1:nau]
    for i in 1:nau
        nb[i] = findall(!iszero, W[i, :])
    end
    return nb
end


"""
    nb_to_adjacency_matrix(nb)

Converts a neighbor-list `nb` (`Vector{Vector{Int}}`) into a dense binary adjacency matrix.
"""
function nb_to_adjacency_matrix(nb)
    nau = length(nb)
    W = zeros(Float64, nau, nau)
    for i in 1:nau
        for neighbor in nb[i]
            if neighbor <= nau
                W[i, neighbor] = 1.0
            end
        end
    end
    return W
end


"""
    nodes(adj)

Extracts undirected edge endpoints `(node1, node2)` and computes the BYM2 scaling factor.
"""
function nodes(adj)
    nau = length(adj)
    node1 = Int[]
    node2 = Int[]
    for i in 1:nau
        for k in adj[i]
            if i < k
                push!(node1, i)
                push!(node2, k)
            end
        end
    end
    e = Edge.(node1, node2)
    g = Graph(e)
    W = Float64.(Graphs.adjacency_matrix(g))
    scalefactor = scaling_factor_bym2(W)
    return node1, node2, scalefactor
end


"""
    get_spatial_graph(centroids, adjacency_edges)

Constructs a `SimpleGraph` from a list of centroids and edge tuples.
"""
function get_spatial_graph(centroids, adjacency_edges)
    n = length(centroids)
    g = SimpleGraph(n)
    centroid_map = Dict(c => i for (i, c) in enumerate(centroids))
    for edge in adjacency_edges
        xi, yi = get(centroid_map, edge[1], 0), get(centroid_map, edge[2], 0)
        if xi > 0 && yi > 0
            add_edge!(g, xi, yi)
        end
    end
    return g
end


"""
    libgeos_lattice_adjacency_matrix(rows::Int, cols::Int; contiguity::Symbol=:queen)

Fast analytical lattice adjacency matrix generation in O(N).
Supports `:queen` (8-neighbor, default) and `:rook` (4-neighbor) contiguity.
"""
function libgeos_lattice_adjacency_matrix(rows::Int, cols::Int; contiguity::Symbol=:queen)
    n = rows * cols
    W = spzeros(Int, n, n)
    idx(r, c) = (r - 1) * cols + c

    for r in 1:rows, c in 1:cols
        u = idx(r, c)
        # Rook neighbors (4-connectivity)
        if r > 1
            W[u, idx(r-1, c)] = 1
        end
        if r < rows
            W[u, idx(r+1, c)] = 1
        end
        if c > 1
            W[u, idx(r, c-1)] = 1
        end
        if c < cols
            W[u, idx(r, c+1)] = 1
        end

        # Queen diagonals (8-connectivity)
        if contiguity == :queen
            if r > 1 && c > 1
                W[u, idx(r-1, c-1)] = 1
            end
            if r > 1 && c < cols
                W[u, idx(r-1, c+1)] = 1
            end
            if r < rows && c > 1
                W[u, idx(r+1, c-1)] = 1
            end
            if r < rows && c < cols
                W[u, idx(r+1, c+1)] = 1
            end
        end
    end
    return W
end


"""
    check_connectivity(g)

Evaluates graph connectivity and returns `(is_connected, n_components, components)`.
"""
function check_connectivity(g)
    comps = connected_components(g)
    return (is_connected = length(comps) == 1, n_components = length(comps), components = comps)
end


"""
    calculate_metrics(au_obj)

Calculates summary density metrics for an areal units object:
`(mean_density, sd_density, cv_density)`.
"""
function calculate_metrics(au_obj)
    pts_mat = hcat([
        [Float64(au_obj.s_x[i]), Float64(au_obj.s_y[i])]
        for i in 1:length(au_obj.s_x)
    ]...)
    c_mat = hcat([[Float64(c[1]), Float64(c[2])] for c in au_obj.centroids]...)
    tree = KDTree(c_mat)
    assigns_knn, _ = knn(tree, pts_mat, 1)
    assignments = [a[1] for a in assigns_knn]
    unit_counts = [count(==(i), assignments) for i in 1:length(au_obj.centroids)]

    valid_entries = filter(x -> !isnan(x) && !ismissing(x), unit_counts)
    if isempty(valid_entries)
        return (mean_density=NaN, sd_density=NaN, cv_density=NaN)
    end
    m_val = mean(valid_entries)
    s_val = std(valid_entries)
    cv_val = s_val / (m_val + 1e-9)

    return (mean_density=m_val, sd_density=s_val, cv_density=cv_val)
end

"""
    extract_graph_edges_and_degrees(g, S::Integer=0)
    -> Tuple{Vector{Tuple{Int, Int}}, Vector{Int}}

Robustly extracts edge index pairs `(src, dst)` and node degree counts from an
adjacency graph (`Graphs.AbstractGraph`) or connectivity matrix (`SparseMatrixCSC` /
`AbstractMatrix`).

# Inputs
- `g`: Graph object (`Graphs.AbstractGraph`) or connectivity matrix (`AbstractMatrix`).
- `S::Integer`: Expected number of units/nodes (default 0, automatically inferred if <= 0).

# Outputs
- `Tuple{Vector{Tuple{Int, Int}}, Vector{Int}}`: Array of `(src, dst)` edge index pairs and
  node degree count vector.
"""
function extract_graph_edges_and_degrees(g, S::Integer = 0)
    if isnothing(g)
        n_units = max(0, Int(S))
        return Tuple{Int, Int}[], zeros(Int, n_units)
    elseif g isa Graphs.AbstractGraph
        n = Graphs.nv(g)
        n_target = S > 0 ? Int(S) : n
        degs = [Graphs.degree(g, i) for i in 1:min(n_target, n)]
        if n_target > n
            append!(degs, zeros(Int, n_target - n))
        end
        edges = [(Graphs.src(e), Graphs.dst(e)) for e in Graphs.edges(g)]
        return edges, degs
    elseif g isa SparseMatrixCSC
        rows = rowvals(g)
        n_cols = size(g, 2)
        n_target = S > 0 ? Int(S) : n_cols
        degs = zeros(Int, n_target)
        edges = Tuple{Int, Int}[]
        for j in 1:min(n_target, n_cols)
            col_nz = nzrange(g, j)
            for idx in col_nz
                i = rows[idx]
                if i <= n_target
                    degs[j] += 1
                    if i > j
                        push!(edges, (j, i))
                    elseif size(g, 1) != size(g, 2) || !issymmetric(g)
                        push!(edges, (j, i))
                    end
                end
            end
        end
        return edges, degs
    elseif g isa AbstractMatrix
        n_rows, n_cols = size(g)
        n_target = S > 0 ? Int(S) : n_cols
        degs = zeros(Int, n_target)
        edges = Tuple{Int, Int}[]
        for j in 1:min(n_target, n_cols)
            for i in 1:min(n_target, n_rows)
                if g[i, j] != 0
                    degs[j] += 1
                    if i > j
                        push!(edges, (j, i))
                    elseif n_rows != n_cols || !issymmetric(g)
                        push!(edges, (j, i))
                    end
                end
            end
        end
        return edges, degs
    end
    throw(ArgumentError(
        "Unsupported graph type: $(typeof(g)). Expected AbstractGraph or AbstractMatrix."
    ))
end

const _extract_graph_edges_and_degrees = extract_graph_edges_and_degrees


