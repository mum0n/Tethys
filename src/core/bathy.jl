"""
    bathymetry.jl

Continuous spatial interpolation, morphological smoothing, and marine grid cell
extraction routines for bathymetric datasets.

All calculations operate directly on in-memory array structures and `GeoDataset` containers.
"""

using Interpolations
using Statistics
using Random

export smooth_bathymetry, get_bathymetry_interpolator, extract_marine_cells,
    sample_marine_cells

"""
    smooth_bathymetry(
        field::AbstractMatrix;
        passes::Integer = 4,
        alpha::Real = 0.5,
        h_min::Real = 0.0
    ) -> Matrix{Float64}

Apply iterative weighted spatial smoothing to a 2D bathymetric elevation field and
enforce an optional minimum water column depth threshold \$h_{\\min}\$.

In each smoothing iteration, interior cell values are updated via a 3x3 kernel:
```math
z_{i,j}^{(k+1)} = (1 - \\alpha) z_{i,j}^{(k)} + \\alpha \\bar{z}_{i,j}
```
where \$\\bar{z}_{i,j}\$ is computed with a center weight of 4 and adjacent weights of 1.
Boundary cells remain unaltered.

# Coordinate and Value Conventions
`field` represents elevation in metres, oriented positive-up (terrestrial elevation \$> 0\$,
water column elevation \$< 0\$, mean sea level \$= 0\$).

When \$h_{\\min} > 0\$, submerged cells (\$z < 0\$) are bounded by \$\\min(z, -h_{\\min})\$.
Subaerial cells (\$z \\ge 0\$) remain unaltered at 0.0 m.

# Arguments
- `field::AbstractMatrix`: 2D elevation grid with dimension ordering `(lon, lat)`.
- `passes::Integer`: Number of smoothing iterations.
- `alpha::Real`: Relaxation coefficient in \$[0, 1]\$.
- `h_min::Real`: Minimum water depth threshold in metres (0.0 disables depth thresholding).

# Returns
- `Matrix{Float64}`: Smoothed elevation grid.
"""
function smooth_bathymetry(
    field::AbstractMatrix;
    passes::Integer = 4,
    alpha::Real = 0.5,
    h_min::Real = 0.0,
    h_max::Real = 0.0
)
    smoothed = Float64.(collect(field))
    n_lon, n_lat = size(smoothed)
    (n_lon < 3 || n_lat < 3) && return smoothed

    for pass in 1:passes
        next = copy(smoothed)
        for j in 2:(n_lat - 1), i in 2:(n_lon - 1)
            block = smoothed[(i - 1):(i + 1), (j - 1):(j + 1)]
            mean = (4.0 * smoothed[i, j] + sum(block) - smoothed[i, j]) / 12.0
            next[i, j] = (1.0 - alpha) * smoothed[i, j] + alpha * mean
        end
        smoothed = next
    end

    if h_min > 0.0
        smoothed = ifelse.(smoothed .< 0, min.(smoothed, -Float64(h_min)), 0.0)
    end
    if h_max > 0.0
        smoothed = ifelse.(smoothed .< -Float64(h_max), -Float64(h_max), smoothed)
    end
    return smoothed
end

"""
    get_bathymetry_interpolator(
        ds::GeoDataset;
        varname::AbstractString = "elevation"
    ) -> Function

Construct a continuous bivariate interpolator `(lon, lat) -> elevation` from a `GeoDataset`.
Domain extrapolation outside source coordinate bounds applies nearest-boundary flat values.

# Arguments
- `ds::GeoDataset`: Bathymetric elevation dataset.
- `varname::AbstractString`: Name of the elevation variable.

# Returns
- `Function`: Anonymous mapping `(lon::Real, lat::Real) -> Float64`.
"""
function get_bathymetry_interpolator(ds::GeoDataset; varname::AbstractString = "elevation")
    elev_var = haskey(ds.variables, varname) ? varname :
               first(variables_like(ds, ["elevation", "altitude", "z", "topo", "bedrock_altitude"]))

    elev_ga = ds.variables[elev_var]
    (lons, _) = axis(ds, :lon)
    (lats, _) = axis(ds, :lat)

    permutation = dim_permutation(elev_ga.dims, :lon, :lat)
    elevation = Array{Float64}(elev_ga.data)
    length(lons) == size(elevation, permutation[1]) || error(
        "Variable $(repr(elev_var)) has size $(size(elevation)) but longitude axis has " *
        "$(length(lons)) elements.")
    length(lats) == size(elevation, permutation[2]) || error(
        "Variable $(repr(elev_var)) has size $(size(elevation)) but latitude axis has " *
        "$(length(lats)) elements.")

    return get_bathymetry_interpolator(permutedims(elevation, permutation), lons, lats)
end

"""
    get_bathymetry_interpolator(
        elevation::AbstractMatrix,
        lon::AbstractVector,
        lat::AbstractVector
    ) -> Function

Construct a continuous bivariate interpolator `(lon, lat) -> elevation` from numeric
grid matrices and coordinate vectors.

# Arguments
- `elevation::AbstractMatrix`: 2D elevation grid structured as `(lon, lat)`.
- `lon::AbstractVector`: Longitude axis coordinates in ascending order.
- `lat::AbstractVector`: Latitude axis coordinates in ascending order.

# Returns
- `Function`: Evaluation function `(lon::Real, lat::Real) -> Float64`.
"""
function get_bathymetry_interpolator(
    elevation::AbstractMatrix,
    lon::AbstractVector,
    lat::AbstractVector
)
    size(elevation) == (length(lon), length(lat)) || error(
        "Elevation grid dimensions $(size(elevation)) must match coordinate lengths " *
        "($(length(lon)), $(length(lat))).")

    itp = Interpolations.interpolate((collect(Float64, lon), collect(Float64, lat)),
                                     Array{Float64}(elevation),
                                     Interpolations.Gridded(Interpolations.Linear()))
    itp_flat = Interpolations.extrapolate(itp, Interpolations.Flat())

    return (lon::Real, lat::Real) -> Float64(itp_flat(Float64(lon), Float64(lat)))
end

"""
    extract_marine_cells(
        ds::GeoDataset;
        elevation_var::AbstractString = "elevation",
        min_seabed_depth::Real = 0.0,
        coastlines::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing
    ) -> NamedTuple

Identify and extract submerged marine grid cells from a bathymetric dataset.

Returns a named tuple `(mask, indices, elevation, lons, lats)` where `mask` is a `BitMatrix`
(with dimension ordering `(lat, lon)`), `indices` contains `CartesianIndex{2}` coordinates
of marine cells, and `elevation`/`lons`/`lats` hold their values.

Submerged cells are filtered by:
1. Seabed elevation criterion: \$z_{\\text{bed}} \\le -\\max(0, h_{\\min})\$.
2. Terrestrial polygon exclusion: cells intersecting closed coastline polygons are masked.

# Arguments
- `ds::GeoDataset`: Bathymetry dataset.
- `elevation_var::AbstractString`: Name of the elevation variable.
- `min_seabed_depth::Real`: Minimum water column depth in metres.
- `coastlines`: Optional collection of closed polygon boundary rings.

# Returns
- `NamedTuple`: Fields `(:mask, :indices, :elevation, :lons, :lats)`.
"""
function extract_marine_cells(
    ds::GeoDataset;
    elevation_var::AbstractString = "elevation",
    min_seabed_depth::Real = 0.0,
    coastlines::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing
)
    ga = ds.variables[elevation_var]
    elev = Float64.(ga.data)

    (pos_lat, pos_lon) = dim_permutation(ga.dims, :lat, :lon)
    if ndims(elev) != 2 || length(unique([pos_lat, pos_lon])) != 2 ||
       sort([pos_lat, pos_lon]) != collect(1:ndims(elev))
        error("Variable $(repr(elevation_var)) has dimension count $(ndims(elev)) with dims " *
              "$([standardize_dimension_name(d.name) for d in ga.dims]). A 2D field " *
              "spanning :lon and :lat is required.")
    end

    (lons, _) = axis(ds, :lon)
    (lats, _) = axis(ds, :lat)
    if length(lons) != size(elev, pos_lon) || length(lats) != size(elev, pos_lat)
        error("Variable $(repr(elevation_var)) dimensions $(size(elev)) do not match " *
              "axis lengths: :lon has $(length(lons)) and :lat has $(length(lats)).")
    end

    elev_t = permutedims(elev, (pos_lat, pos_lon))

    p_lon = sortperm(lons)
    p_lat = sortperm(lats)
    lons = lons[p_lon]
    lats = lats[p_lat]
    elev_t = elev_t[p_lat, p_lon]

    n_lon, n_lat = length(lons), length(lats)

    h_threshold = -max(0.0, Float64(min_seabed_depth))
    mask = BitMatrix(elev_t .<= h_threshold)

    if coastlines !== nothing
        _exclude_land!(mask, lons, lats, coastlines)
    end

    marine_indices = CartesianIndex{2}[]
    marine_elev = Float64[]
    marine_lons = Float64[]
    marine_lats = Float64[]
    for j in 1:n_lat, i in 1:n_lon
        if mask[j, i]
            push!(marine_indices, CartesianIndex(j, i))
            push!(marine_elev, elev_t[j, i])
            push!(marine_lons, lons[i])
            push!(marine_lats, lats[j])
        end
    end

    return (
        mask = mask,
        indices = marine_indices,
        elevation = marine_elev,
        lons = marine_lons,
        lats = marine_lats,
    )
end

"""
    _exclude_land!(mask, lons, lats, coastlines) -> BitMatrix

Mask grid cells inside terrestrial boundary polygon rings in place.
"""
function _exclude_land!(
    mask::BitMatrix,
    lons::Vector{Float64},
    lats::Vector{Float64},
    coastlines::AbstractVector{<:NamedTuple}
)
    for poly in coastlines
        (poly.lons[begin] == poly.lons[end] && poly.lats[begin] == poly.lats[end]) || continue
        i0 = max(1, searchsortedfirst(lons, minimum(poly.lons)))
        i1 = min(length(lons), searchsortedlast(lons, maximum(poly.lons)))
        j0 = max(1, searchsortedfirst(lats, minimum(poly.lats)))
        j1 = min(length(lats), searchsortedlast(lats, maximum(poly.lats)))
        for j in j0:j1, i in i0:i1
            if mask[j, i] && point_in_polygon(lons[i], lats[j], poly.lons, poly.lats)
                mask[j, i] = false
            end
        end
    end
    return mask
end

"""
    sample_marine_cells(
        n_samples::Int;
        bathymetry_ds::GeoDataset,
        elevation_var::AbstractString = "elevation",
        min_seabed_depth::Real = 0.0,
        jitter_scale::Real = 0.5,
        rng::AbstractRNG = Random.default_rng(),
        coastlines::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing
    ) -> NamedTuple{(:lons, :lats, :depths), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}

Sample `n_samples` spatial locations from submerged marine cells weighted proportionally
to absolute bathymetric depth \$|z_{\\text{bed}}|\$, with uniform intra-cell perturbation.

Sampling follows an inverse transform sampling approach over normalized cumulative cell
weights. Intra-cell coordinates are perturbed within \$\\pm 0.5 \\times \\text{jitter\\_scale} \\times \\Delta\$.

# Arguments
- `n_samples::Int`: Number of spatial coordinates to sample.
- `bathymetry_ds::GeoDataset`: Bathymetry elevation dataset.
- `elevation_var::AbstractString`: Name of the elevation variable.
- `min_seabed_depth::Real`: Minimum depth threshold in metres.
- `jitter_scale::Real`: Perturbation scale factor in \$[0, 1]\$.
- `rng::AbstractRNG`: Random number generator.
- `coastlines`: Optional collection of coastline boundary polygons.

# Returns
- `NamedTuple`: Fields `(:lons, :lats, :depths)`.
"""
function sample_marine_cells(
    n_samples::Int;
    bathymetry_ds::GeoDataset,
    elevation_var::AbstractString = "elevation",
    min_seabed_depth::Real = 0.0,
    jitter_scale::Real = 0.5,
    rng::AbstractRNG = Random.default_rng(),
    coastlines::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing
)
    marine = extract_marine_cells(
        bathymetry_ds;
        elevation_var = elevation_var,
        min_seabed_depth = min_seabed_depth,
        coastlines = coastlines
    )
    isempty(marine.indices) && error(
        "No marine cells found matching elevation variable '$(elevation_var)' " *
        "and minimum depth threshold $(min_seabed_depth) m.")

    weights = abs.(marine.elevation)
    weights_sum = sum(weights)
    if weights_sum == 0.0
        weights = ones(length(marine.elevation))
        weights_sum = length(marine.elevation)
    end
    cum_weights = cumsum(weights ./ weights_sum)

    lons = marine.lons
    lats = marine.lats
    d_lon = _axis_spacing(lons)
    d_lat = _axis_spacing(lats)

    lons_out = Float64[]
    lats_out = Float64[]
    depths_out = Float64[]

    n_cells = length(marine.elevation)
    for _ in 1:n_samples
        u = rand(rng)
        idx = searchsortedfirst(cum_weights, u)
        if idx > length(cum_weights)
            idx = length(cum_weights)
        end
        if idx < 1 || idx > n_cells
            error("Sampled linear index $idx is outside valid range [1, $n_cells].")
        end

        lon = marine.lons[idx] + (2 * rand(rng) - 1) * 0.5 * jitter_scale * d_lon
        lat = marine.lats[idx] + (2 * rand(rng) - 1) * 0.5 * jitter_scale * d_lat
        push!(lons_out, lon)
        push!(lats_out, lat)
        push!(depths_out, marine.elevation[idx])
    end

    return (lons = lons_out, lats = lats_out, depths = depths_out)
end

"""
    _axis_spacing(coords::Vector{Float64}) -> Float64

Compute the minimum non-zero step between adjacent coordinate values.
Returns 1.0 if fewer than two distinct coordinate values are present.
"""
function _axis_spacing(coords::Vector{Float64})
    length(coords) < 2 && return 1.0
    gaps = abs.(diff(sort(coords)))
    gaps = gaps[gaps .> 0]
    return isempty(gaps) ? 1.0 : minimum(gaps)
end
