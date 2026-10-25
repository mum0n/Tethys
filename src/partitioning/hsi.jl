# -----------------------------------------------------------------------------
# hsi.jl -- (GeoPartitioning) extracted verbatim from TethysBayes src/partitioning.jl
# lines 2780-3039 (HSI unit interpolation and field resharding).
# ------------------------------------------------------------------------------
"""
    _interpolate_hsi(hsi_2d, years, s_idx, decimal_year; ref_doy=244.0)
    -> Float64

Linear interpolation of the annual posterior-mean HSI for spatial unit `s_idx`
at continuous time `decimal_year`.

The annual predictions are anchored at `ref_doy` (day of year; 244 ≈ Sept 1):

    t_ref = (ref_doy − 1) / 365.25
    u     = decimal_year − t_ref
    y1    = clamp(⌊u⌋, y_min, y_max − 1)
    α     = u − y1                          ∈ [0, 1)
    HSI   = (1 − α) HSI(s, y1) + α HSI(s, y1+1)

# Arguments
- `hsi_2d`: Posterior-mean HSI matrix (S × T).
- `years`: Annual year labels (e.g. 1999:2025).
- `s_idx`: Spatial unit index (1-based).
- `decimal_year`: Continuous time in decimal years.
- `ref_doy`: Reference survey day of year (default 244.0).

# Returns
- `Float64`: Interpolated HSI clamped to [0, 1].
"""
function _interpolate_hsi(
    hsi_2d::AbstractMatrix{Float64},
    years::AbstractVector{<:Integer},
    s_idx::Integer,
    decimal_year::Real;
    ref_doy::Real = 244.0
)::Float64
    y_min = first(years)
    y_max = last(years)
    t_off = (ref_doy - 1.0) / 365.25
    u     = Float64(decimal_year) - t_off
    u_cl  = clamp(u, Float64(y_min), Float64(y_max))
    y1    = clamp(floor(Int, u_cl), y_min, y_max - 1)
    α     = u_cl - Float64(y1)
    t1    = clamp(y1 - y_min + 1, 1, length(years))
    t2    = clamp(t1 + 1, 1, length(years))
    return clamp((1.0 - α) * hsi_2d[s_idx, t1] + α * hsi_2d[s_idx, t2], 0.0, 1.0)
end


"""
    build_monthly_hsi_matrix(
        hsi_mean::AbstractMatrix{Float64},
        years::AbstractVector{<:Integer};
        ref_doy::Real = 244.0
    ) -> Tuple{Matrix{Float64}, Dict{Tuple{Int, Int}, Int}}

Discretizes multi-year spatial Habitat Suitability Index (HSI) surfaces onto a
regular monthly grid (12 calendar months per survey year) via continuous
piecewise-linear temporal interpolation, evaluating each cell at mid-month (day 15)
with leap-year awareness.

# Mathematical Formulation
Given annual HSI values ``h_s(y)`` for spatial unit ``s`` in year ``y``, the fractional
calendar year at day-of-year ``d`` of year ``y`` is:
```math
t = y + \\frac{d - 1}{D(y)}
```
where ``D(y) \\in \\{365, 366\\}``. Anchored to annual survey reference day
``d_{\\text{ref}}``, the mid-month HSI is linearly interpolated between adjacent survey epochs:
```math
h_s(t) = (1 - \\alpha) h_s(y_1) + \\alpha h_s(y_2)
```
clamped to ``[0, 1]``.

# Arguments
- `hsi_mean`: Spatial HSI posterior mean matrix (size ``S \\times T_{\\text{years}}``).
- `years`: Integer vector of survey years of length ``T_{\\text{years}}``.
- `ref_doy`: Reference survey day of year (default `244.0` = September 1).

# Returns
- `Tuple{Matrix{Float64}, Dict{Tuple{Int, Int}, Int}}`:
  - `mat`: Monthly discretized HSI matrix (size ``S \\times 12 T_{\\text{years}}``).
  - `lookup`: Dictionary mapping `(year, month)` tuples to matrix column indices.
"""
function build_monthly_hsi_matrix(
    hsi_mean::AbstractMatrix{Float64},
    years::AbstractVector{<:Integer};
    ref_doy::Real = 244.0
)::Tuple{Matrix{Float64}, Dict{Tuple{Int, Int}, Int}}

    S      = size(hsi_mean, 1)
    n_cols = length(years) * 12
    
    # Use undef since we overwrite every element (avoids zeroing overhead)
    mat    = Matrix{Float64}(undef, S, n_cols)
    
    # Pre-size the dictionary to avoid reallocations
    lookup = Dict{Tuple{Int, Int}, Int}()
    sizehint!(lookup, n_cols)
    
    col = 1
    for yr in years
        days_in_yr = Dates.isleapyear(yr) ? 366.0 : 365.0
        for m in 1:12
            doy_mid = Float64(Dates.dayofyear(Date(yr, m, 15)))
            dec_yr  = Float64(yr) + (doy_mid - 1.0) / days_in_yr
            
            # @inbounds safely removes bounds checking for peak speed
            @inbounds for s in 1:S
                mat[s, col] = _interpolate_hsi(
                    hsi_mean, years, s, dec_yr; ref_doy=ref_doy)
            end
            
            lookup[(yr, m)] = col
            col += 1
        end
    end
    
    return mat, lookup
end


"""
    load_hsi_jld2(path; hsi_key="hsi", years_key="years",
                  auids_key="auids", ref_doy=244.0) -> NamedTuple

Load a JLD2 bundle containing a 3D posterior HSI array and compute derived
summaries needed for movement modelling.

# Expected JLD2 keys
- `hsi`:   Array{Float64, 3} (S × T × N_draws).
- `years`: Vector{Int} of length T.
- `auids`: (optional) spatial unit identifiers.

# Arguments
- `path`: Path to the JLD2 file.
- `hsi_key`, `years_key`, `auids_key`: JLD2 key names.
- `ref_doy`: Reference day of year (default 244.0 = Sept 1).

# Returns
`NamedTuple`:
- `hsi`, `years`, `auids`.
- `hsi_mean` (S × T): posterior mean.
- `hsi_sd` (S × T): posterior standard deviation.
- `hsi_spatial_mean` (length S): multi-year mean per spatial unit.
- `monthly_hsi` (S × 12T): monthly discretization.
- `month_lookup`: Dict{(year, month) => column_index}.
"""
function load_hsi_jld2(
    path::AbstractString;
    hsi_key::AbstractString   = "hsi",
    years_key::AbstractString = "years",
    auids_key::AbstractString = "auids",
    ref_doy::Real             = 244.0
)::NamedTuple
    isfile(path) || error("HSI JLD2 not found: $(path)")
    bundle = JLD2.load(path)
    actual_hsi_key = if haskey(bundle, hsi_key)
        hsi_key
    elseif haskey(bundle, "sims")
        "sims"
    elseif haskey(bundle, "predictions")
        "predictions"
    else
        hsi_key
    end
    hsi    = bundle[actual_hsi_key]
    years  = bundle[years_key]
    auids  = haskey(bundle, auids_key) ? bundle[auids_key] : collect(1:size(hsi, 1))

    hsi_mean          = dropdims(mean(hsi, dims=3), dims=3)
    hsi_sd            = dropdims(std(hsi, dims=3),  dims=3)
    hsi_spatial_mean  = vec(mean(hsi_mean, dims=2))

    monthly_hsi, month_lookup = build_monthly_hsi_matrix(
        Float64.(hsi_mean), years; ref_doy=ref_doy)

    return (
        hsi              = hsi,
        years            = years,
        auids            = auids,
        hsi_mean         = hsi_mean,
        hsi_sd           = hsi_sd,
        hsi_spatial_mean = hsi_spatial_mean,
        monthly_hsi      = monthly_hsi,
        month_lookup     = month_lookup
    )
end
"""
    reshard_hsi_field(
        src_vals, src_coords_km, dest_centroids_km; k=6
    ) -> Vector{Float64}

Reshard a scalar field from source centroids to destination centroids using
planar inverse-distance-squared weighting (IDW) among the `k` nearest source
points.

All coordinates must be in the same planar km system.

Weight: w_j = 1/d_j² (d_j < 1e-9 → w = 1e18, i.e. coincident → snap).

# Arguments
- `src_vals`: Values at source centroids (length S_src).
- `src_coords_km`: Planar km coordinates of source centroids.
- `dest_centroids_km`: Planar km coordinates of destination centroids.
- `k`: Nearest neighbours to use (default 6; capped at S_src).

# Returns
- `Vector{Float64}` of length S_dest.
"""
function reshard_hsi_field(
    src_vals::AbstractVector{<:Real},
    src_coords_km::Vector{Tuple{Float64, Float64}},
    dest_centroids_km::Vector{Tuple{Float64, Float64}};
    k::Integer = 6
)::Vector{Float64}

    N_src = length(src_coords_km)
    N_dest = length(dest_centroids_km)
    k_eff = min(Int(k), length(src_vals))

    # 1. Preallocate and fill matrices efficiently (no splatting)
    src_mat = Matrix{Float64}(undef, 2, N_src)
    for i in 1:N_src
        src_mat[1, i] = src_coords_km[i][1]
        src_mat[2, i] = src_coords_km[i][2]
    end

    dest_mat = Matrix{Float64}(undef, 2, N_dest)
    for i in 1:N_dest
        dest_mat[1, i] = dest_centroids_km[i][1]
        dest_mat[2, i] = dest_centroids_km[i][2]
    end

    # 2. Build tree and find neighbors
    tree = KDTree(src_mat)
    idxs, dists = knn(tree, dest_mat, k_eff)

    # 3. Allocation-free inner loop for IDW
    out = Vector{Float64}(undef, N_dest)
    
    @inbounds for j in 1:N_dest
        w_sum = 0.0
        val_sum = 0.0
        
        # Loop over the k neighbors for the j-th destination point
        for i in 1:k_eff
            idx = idxs[j][i]
            d   = dists[j][i]
            
            # d * d is slightly faster than d^2
            w = d < 1e-9 ? 1e18 : 1.0 / (d * d)
            
            val_sum += w * src_vals[idx]
            w_sum   += w
        end
        
        # Protect against division by zero (though mathematically impossible with strictly positive w)
        out[j] = w_sum > 0.0 ? (val_sum / w_sum) : 0.0 
    end
    
    return out
end

