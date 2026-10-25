# -----------------------------------------------------------------------------
# temporal.jl -- (GeoPartitioning) extracted verbatim from TethysBayes src/partitioning.jl
# lines 2058-2367 (Section 5: Temporal Discretization, Spatiotemporal Units, Cross-Validation).
# ------------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Section 5: Temporal Discretization, Spatiotemporal Units & Cross-Validation
# -----------------------------------------------------------------------------

"""
    discretize_data(X; method="quantile", N_cat=9, brks=nothing, ...)

Discretizes a continuous variable `X` into `N_cat` bins using:
- `"quantile"`: Empirical quantiles.
- `"regular"`: Equal-width bins across the data extent.
- `"quantile_regular"`: Equal-width bins between lower/upper quantiles.
- `"kmeans"`: 1D k-means clustering.
- `"jenks"`: Jenks Natural Breaks optimization.
- `"provided"`: Precomputed explicit break bounds.

Returns `(idx = idx, brks = collect(brks), mids = mids, N_cat = N_cat)`.
"""
function discretize_data(
    X; method="quantile", N_cat=9, brks=nothing, probs=nothing, dx=nothing,
    minv=nothing, maxv=nothing, quantile_bounds=[0.025, 0.975]
)
    if method == "quantile"
        probs = isnothing(probs) ? range(0, stop=1, length=N_cat+1) : probs
        brks = quantile(X, probs)
        brks[end] = brks[end] + 1e-6
    elseif method == "regular"
        minv = isnothing(minv) ? minimum(X) : minv
        maxv = isnothing(maxv) ? maximum(X) : maxv
        dx = isnothing(dx) ? (maxv - minv) / N_cat : dx
        brks = minv:dx:maxv
    elseif method == "quantile_regular"
        q = quantile(X, quantile_bounds)
        brks = range(q[1], q[2], length=N_cat + 1)
    elseif method == "kmeans"
        X_mat = reshape(X, 1, :)
        R = kmeans(X_mat, N_cat)
        centers = sort(vec(R.centers))
        brks = vcat(minimum(X), (centers[1:end-1] + centers[2:end]) / 2.0, maximum(X))
    elseif method == "jenks"
        brks = _jenks_breaks(X, N_cat)
    elseif method == "provided"
        if isnothing(brks)
            error("Method 'provided' requires the 'brks' argument.")
        end
        N_cat = length(brks) - 1
    else
        error("Discretization method '$method' not recognized.")
    end

    function get_idx(x::Real, breaks, n_categories)
        raw_idx = searchsortedfirst(breaks, x) - 1
        return clamp(raw_idx, 1, n_categories)
    end

    idx = map(x -> get_idx(x, brks, N_cat), X)
    mids = brks[1:end-1] .+ diff(brks) ./ 2.0

    return (idx = idx, brks = collect(brks), mids = mids, N_cat = N_cat)
end


function _jenks_breaks(data::AbstractVector{<:Real}, n_classes::Int)
    s_data = sort(data)
    n_data = length(s_data)
    lower_class_limits = zeros(n_data, n_classes)
    variance_combinations = zeros(n_data, n_classes)

    for i in 1:n_data
        lower_class_limits[i, 1] = 1
        variance_combinations[i, 1] = sum((s_data[1:i] .- mean(s_data[1:i])).^2)
    end

    for k in 2:n_classes
        for i in k:n_data
            min_variance = Inf
            best_break = 0
            for j in k:i
                ssd = sum((s_data[j:i] .- mean(s_data[j:i])).^2)
                total_variance = ssd + variance_combinations[j-1, k-1]
                if total_variance < min_variance
                    min_variance = total_variance
                    best_break = j
                end
            end
            lower_class_limits[i, k] = best_break
            variance_combinations[i, k] = min_variance
        end
    end

    breaks = zeros(n_classes + 1)
    breaks[n_classes + 1] = s_data[end]
    breaks[1] = s_data[1]
    for k in n_classes:-1:2
        break_index = Int(lower_class_limits[n_data, k])
        breaks[k] = s_data[break_index]
        n_data = break_index - 1
    end

    return breaks
end


"""
    assign_time_units(t_v; time_method="quantile_regular", t_N=nothing, u_N=nothing, kwargs...)

Discretizes a time vector into categorical unit indices. Handles continuous floats and
  discrete integers.
"""
function assign_time_units(
    t_v::AbstractVector{<:Real};
    time_method="quantile_regular", t_N=nothing, u_N=nothing, kwargs...
)
    local_t_N = isnothing(t_N) ? (isnothing(u_N) ? 10 : u_N) : t_N
    return discretize_data(t_v; method=time_method, N_cat=local_t_N, kwargs...)
end

function assign_time_units(
    t_v::AbstractVector{<:Integer}; time_method="unique", t_N=nothing, u_N=nothing, kwargs...
)
    unique_times = sort(unique(t_v))
    N_cat = length(unique_times)
    brks = if N_cat > 1
        vcat(
    [unique_times[1] - 0.5],
    (unique_times[1:end-1] + unique_times[2:end]) / 2.0,
    [unique_times[end] + 0.5]
)
    elseif N_cat == 1
        [unique_times[1] - 0.5, unique_times[1] + 0.5]
    else
        Float64[]
    end
    val_to_idx = Dict(val => i for (i, val) in enumerate(unique_times))
    idx = [val_to_idx[v] for v in t_v]
    mids = Float64.(unique_times)

    return (idx=idx, brks=brks, mids=mids, N_cat=N_cat)
end

"""
    assign_time_units(t_v::AbstractVector; time_method="unique", t_N=nothing, u_N=nothing, kwargs...)

Discretizes arbitrary temporal identifiers (dates, strings, symbols, or general objects) into
consecutive 1-based indices `1:N_cat` based on their sorted unique values.
"""
function assign_time_units(
    t_v::AbstractVector; time_method="unique", t_N=nothing, u_N=nothing, kwargs...
)
    unique_times = sort(unique(t_v))
    N_cat = length(unique_times)
    brks = collect(1:(N_cat + 1))
    val_to_idx = Dict(val => i for (i, val) in enumerate(unique_times))
    idx = [val_to_idx[v] for v in t_v]
    mids = unique_times

    return (idx=idx, brks=brks, mids=mids, N_cat=N_cat)
end


"""
    assign_spatiotemporal_units(df::DataFrame; space_x=nothing, space_y=nothing,
                                time_var=nothing, target_space_col=nothing,
                                target_time_col=nothing, area_method=:avt,
                                target_units=10, time_method="unique", t_N=nothing, kwargs...)

Simultaneously discretizes space and time, producing synchronized `(s_idx, t_idx, st_idx)`
and dimensional metadata for spatiotemporal modeling. Automatically detects coordinate
and temporal variables if not explicitly specified.
"""
function assign_spatiotemporal_units(
    df::DataFrame;
    space_x::Union{Symbol, AbstractString, Nothing}=nothing,
    space_y::Union{Symbol, AbstractString, Nothing}=nothing,
    time_var::Union{Symbol, AbstractString, Nothing}=nothing,
    target_space_col::Union{Symbol, AbstractString, Nothing}=nothing,
    target_time_col::Union{Symbol, AbstractString, Nothing}=nothing,
    area_method::Symbol=:avt,
    target_units::Integer=10,
    time_method="unique",
    t_N=nothing,
    kwargs...
)
    col_x, col_y = _detect_xy_columns(df; x=space_x, y=space_y)
    col_t = _detect_time_column(df; time_var=time_var)

    au_space = assign_spatial_units(
        df; x=col_x, y=col_y, area_method=area_method,
        target_units=target_units, kwargs...
    )
    au_time = assign_time_units(df[!, col_t]; time_method=time_method, t_N=t_N, kwargs...)

    S = length(au_space.centroids)
    T = au_time.N_cat
    s_idx = au_space.s_idx
    t_idx = au_time.idx
    st_idx = (t_idx .- 1) .* S .+ s_idx

    scalefactor_s = scaling_factor_bym2(au_space.W)

    if !isnothing(target_space_col)
        df[!, Symbol(target_space_col)] = s_idx
    end
    if !isnothing(target_time_col)
        df[!, Symbol(target_time_col)] = t_idx
    end

    return (
        au_spatial = au_space,
        au_temporal = au_time,
        s_idx = s_idx,
        t_idx = t_idx,
        st_idx = st_idx,
        S = S,
        T = T,
        ST = S * T,
        W_spatial = au_space.W,
        scaling_factor_spatial = scalefactor_s
    )
end


"""
    spatial_block_cv(s_x, s_y; n_folds::Integer=5, method::Symbol=:kmeans)

Generates spatial cross-validation fold assignments (`1:n_folds`) to assess model generalizability
without spatial autocorrelation leakage across training/test splits.
"""
function spatial_block_cv(
    s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real};
    n_folds::Integer=5, method::Symbol=:kmeans
)
    n = length(s_x)
    if n == 0
        return Int[]
    elseif n < n_folds
        return collect(1:n)
    end

    if method == :kmeans
        pts_mat = hcat([[Float64(s_x[i]), Float64(s_y[i])] for i in 1:n]...)
        R = kmeans(pts_mat, n_folds)
        return R.assignments
    elseif method == :grid
        xmin, xmax = extrema(s_x)
        ymin, ymax = extrema(s_y)
        side = ceil(Int, sqrt(n_folds))
        x_bins = range(xmin, stop=xmax + 1e-6, length=side + 1)
        y_bins = range(ymin, stop=ymax + 1e-6, length=side + 1)
        folds = zeros(Int, n)
        for i in 1:n
            bx = clamp(searchsortedfirst(x_bins, s_x[i]) - 1, 1, side)
            by = clamp(searchsortedfirst(y_bins, s_y[i]) - 1, 1, side)
            cell_id = (by - 1) * side + bx
            folds[i] = ((cell_id - 1) % n_folds) + 1
        end
        return folds
    else
        error("Unknown spatial block CV method: $method. Choose :kmeans or :grid.")
    end
end

spatial_block_cv(coords::Vector{<:Tuple{Real, Real}}; kwargs...) =
    spatial_block_cv([p[1] for p in coords], [p[2] for p in coords]; kwargs...)


"""
    estimate_local_kde_with_extrapolation(s_coord_tuple, t_idx, target_ts; grid_res=600,
      sd_extension_factor=0.25)

Computes 2D Gaussian Kernel Density Estimation (KDE) on a regular grid for a specific time slice.
"""
function estimate_local_kde_with_extrapolation(
    s_coord_tuple, t_idx, target_ts; grid_res=600, sd_extension_factor=0.25
)
    filtered_pts = [p for (i, p) in enumerate(s_coord_tuple) if t_idx[i] == target_ts]
    if isempty(filtered_pts)
        error("No points found for target time slice $target_ts.")
    end
    xs = [p[1] for p in filtered_pts]
    ys = [p[2] for p in filtered_pts]

    bw_x = max(std(xs) * sd_extension_factor, 1e-4)
    bw_y = max(std(ys) * sd_extension_factor, 1e-4)

    x_min, x_max = minimum(xs) - bw_x, maximum(xs) + bw_x
    y_min, y_max = minimum(ys) - bw_y, maximum(ys) + bw_y

    x_grid = collect(range(x_min, stop=x_max, length=grid_res))
    y_grid = collect(range(y_min, stop=y_max, length=grid_res))
    intensity = zeros(grid_res, grid_res)

    for i in 1:grid_res
        x_val = x_grid[i]
        for j in 1:grid_res
            y_val = y_grid[j]
            for (px, py) in filtered_pts
                dx = (x_val - px) / bw_x
                dy = (y_val - py) / bw_y
                intensity[i, j] += exp(-0.5 * (dx^2 + dy^2))
            end
        end
    end

    intensity ./= max(sum(intensity), 1e-9)
    return x_grid, y_grid, intensity
end




