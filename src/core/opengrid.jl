"""
    opengrid.jl

Bilinear and trilinear grid-to-grid field interpolation and spatial buffer domain
expansion routines for regular Cartesian and geographic coordinates.
"""

import Interpolations: interpolate, extrapolate, Gridded, Linear, Flat

export regrid_2d_field, regrid_3d_field, buffer_distance_to_degrees,
    expand_domain_with_buffer

"""
    regrid_2d_field(
        src_lon::AbstractVector,
        src_lat::AbstractVector,
        src_data::AbstractMatrix,
        target_lon::AbstractVector,
        target_lat::AbstractVector
    ) -> Matrix{Float64}

Interpolate a 2D scalar field across regular coordinate axes using bilinear interpolation
with flat boundary extrapolation.

# Arguments
- `src_lon::AbstractVector`: Source longitude coordinates.
- `src_lat::AbstractVector`: Source latitude coordinates.
- `src_data::AbstractMatrix`: Source field matrix with dimensions `(lon, lat)` or `(lat, lon)`.
- `target_lon::AbstractVector`: Target longitude coordinates.
- `target_lat::AbstractVector`: Target latitude coordinates.

# Returns
- `Matrix{Float64}`: Interpolated field of size `(length(target_lon), length(target_lat))`.
"""
function regrid_2d_field(
    src_lon::AbstractVector,
    src_lat::AbstractVector,
    src_data::AbstractMatrix,
    target_lon::AbstractVector,
    target_lat::AbstractVector
)
    lons = Float64.(collect(src_lon))
    lats = Float64.(collect(src_lat))
    size(lons, 1) * size(lats, 1) == length(src_data) || error(
        "Source field dimension $(size(src_data)) does not match axis lengths " *
        "$(length(lons)) x $(length(lats)).")

    field = Array{Float64}(src_data)
    if size(field) == (length(lats), length(lons))
        field = permutedims(field, (2, 1))
    end

    if !issorted(lons)
        p = sortperm(lons)
        lons = lons[p]
        field = field[p, :]
    end
    if !issorted(lats)
        p = sortperm(lats)
        lats = lats[p]
        field = field[:, p]
    end

    itp = interpolate((lons, lats), field, Gridded(Linear()))
    itp_flat = extrapolate(itp, Flat())

    out = Matrix{Float64}(undef, length(target_lon), length(target_lat))
    for (i, x) in enumerate(target_lon)
        for (j, y) in enumerate(target_lat)
            out[i, j] = itp_flat(Float64(x), Float64(y))
        end
    end
    return out
end

"""
    regrid_2d_field(
        src_ds::GeoDataset,
        src_var::String,
        target_lon::AbstractVector,
        target_lat::AbstractVector
    ) -> Matrix{Float64}

Regrid a 2-D variable from a dataset by resolving its axes and delegating to the
raw-array method above.
"""
function regrid_2d_field(
    src_ds::GeoDataset,
    src_var::String,
    target_lon::AbstractVector,
    target_lat::AbstractVector
)
    haskey(src_ds.variables, src_var) || error("Variable $src_var not found in dataset")
    src_ga = src_ds.variables[src_var]
    src_lons, _ = axis(src_ds, :lon)
    src_lats, _ = axis(src_ds, :lat)
    return regrid_2d_field(src_lons, src_lats, src_ga.data, target_lon, target_lat)
end

"""
    regrid_3d_field(
        src_ds::GeoDataset,
        src_var::String,
        target_lon::AbstractVector,
        target_lat::AbstractVector,
        target_depth::AbstractVector
    ) -> Array{Float64, 3}

Interpolate a 3D field variable `(lon, lat, depth)` onto target coordinate axes
using trilinear interpolation with flat boundary extrapolation.

# Arguments
- `src_ds::GeoDataset`: Source dataset containing `src_var`.
- `src_var::String`: Variable identifier to interpolate.
- `target_lon::AbstractVector`: Target longitude coordinates.
- `target_lat::AbstractVector`: Target latitude coordinates.
- `target_depth::AbstractVector`: Target depth coordinates.

# Returns
- `Array{Float64, 3}`: 3D interpolated array of size
  `(length(target_lon), length(target_lat), length(target_depth))`.
"""
function regrid_3d_field(
    src_ds::GeoDataset,
    src_var::String,
    target_lon::AbstractVector,
    target_lat::AbstractVector,
    target_depth::AbstractVector
)
    haskey(src_ds.variables, src_var) || error("Variable $src_var not found in dataset")

    src_ga = src_ds.variables[src_var]
    src_data = src_ga.data

    src_lons, _ = axis(src_ds, :lon)
    src_lats, _ = axis(src_ds, :lat)
    src_deps, _ = axis(src_ds, :depth)

    if ndims(src_data) == 3
        if size(src_data, 1) == length(src_lats) && size(src_data, 2) == length(src_lons)
            src_data = permutedims(src_data, (2, 1, 3))
        end
    end

    if !issorted(src_lons)
        p = sortperm(src_lons)
        src_lons = src_lons[p]
        src_data = src_data[p, :, :]
    end
    if !issorted(src_lats)
        p = sortperm(src_lats)
        src_lats = src_lats[p]
        src_data = src_data[:, p, :]
    end
    if !issorted(src_deps)
        p = sortperm(src_deps)
        src_deps = src_deps[p]
        src_data = src_data[:, :, p]
    end

    itp = interpolate((src_lons, src_lats, src_deps), src_data, Gridded(Linear()))
    itp_flat = extrapolate(itp, Flat())

    n_tgt_x = length(target_lon)
    n_tgt_y = length(target_lat)
    n_tgt_z = length(target_depth)
    interpolated = Array{Float64}(undef, n_tgt_x, n_tgt_y, n_tgt_z)

    for (i_idx, x_val) in enumerate(target_lon)
        for (j_idx, y_val) in enumerate(target_lat)
            for (k_idx, z_val) in enumerate(target_depth)
                interpolated[i_idx, j_idx, k_idx] = itp_flat(x_val, y_val, z_val)
            end
        end
    end

    return interpolated
end

"""
    buffer_distance_to_degrees(
        buffer_km::Real,
        ref_lat::Real = 44.5
    ) -> Tuple{Float64, Float64}

Convert a linear physical buffer distance in kilometers \$d_{\\text{buf}}\$ into
equivalent geographic longitude and latitude degree increments \$(\\Delta\\lambda, \\Delta\\phi)\$
at a specified reference latitude \$\\phi_0\$.

# Mathematical Formulation
Using the spherical Earth model with mean radius \$R_{\\text{earth}} = 6371.0088\\text{ km}\$:
```math
\\Delta\\phi = \\frac{d_{\\text{buf}}}{R_{\\text{earth}}} \\times \\left(\\frac{180^\\circ}{\\pi}\\right)
```
```math
\\Delta\\lambda = \\frac{d_{\\text{buf}}}{R_{\\text{earth}} \\cos(\\deg2rad(\\phi_0))} \\times \\left(\\frac{180^\\circ}{\\pi}\\right)
```

# Inputs
- `buffer_km::Real`: Buffer distance in kilometers (e.g. 100.0 km).
- `ref_lat::Real`: Reference latitude in degrees North (default 44.5°N).

# Outputs
- `Tuple{Float64, Float64}`: `(dlon, dlat)` degree increments.
"""
function buffer_distance_to_degrees(
    buffer_km::Real,
    ref_lat::Real = 44.5
)
    if buffer_km < 0.0
        error("Buffer distance must be non-negative: $(buffer_km) km")
    end

    lat = Float64(ref_lat)
    if lat < -90.0 || lat > 90.0
        error("Reference latitude must lie in [-90.0, 90.0] degrees: $(ref_lat)")
    end

    r_earth_km = 6371.0088
    km_per_deg_lat = (π * r_earth_km) / 180.0
    dlat = Float64(buffer_km) / km_per_deg_lat

    cos_lat = cos(deg2rad(lat))
    km_per_deg_lon = km_per_deg_lat * max(cosd(85.0), cos_lat)
    dlon = Float64(buffer_km) / km_per_deg_lon

    return (dlon, dlat)
end

"""
    expand_domain_with_buffer(
        lon_range::Tuple{Real, Real},
        lat_range::Tuple{Real, Real};
        buffer_km::Real = 100.0
    ) -> Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}

Expand a geographic bounding box \$(\\lambda_{\\min}, \\lambda_{\\max}) \\times (\\phi_{\\min}, \\phi_{\\max})\$
outward by a user-defined physical buffer distance (default 100.0 km).

# Inputs
- `lon_range::Tuple{Real, Real}`: Input longitude bounds.
- `lat_range::Tuple{Real, Real}`: Input latitude bounds.
- `buffer_km::Real`: Buffer distance in kilometers (default 100.0 km).

# Outputs
- `Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}`: `(buffered_lon_range, buffered_lat_range)`
"""
function expand_domain_with_buffer(
    lon_range::Tuple{Real, Real},
    lat_range::Tuple{Real, Real};
    buffer_km::Real = 100.0
)
    ref_lat = 0.5 * (Float64(lat_range[1]) + Float64(lat_range[2]))
    dlon, dlat = buffer_distance_to_degrees(buffer_km, ref_lat)

    buf_lon = (
        max(-180.0, Float64(lon_range[1]) - dlon),
        min(180.0, Float64(lon_range[2]) + dlon)
    )
    buf_lat = (
        max(-90.0, Float64(lat_range[1]) - dlat),
        min(90.0, Float64(lat_range[2]) + dlat)
    )

    return (buf_lon, buf_lat)
end
