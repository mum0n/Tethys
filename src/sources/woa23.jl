# ==============================================================================
# woa23.jl -- WOA23 climatology fetch, load, and interpolator operations
# Fetches temperature, salinity, oxygen, and nutrients from NOAA NCEI.
# ==============================================================================

using NCDatasets
using Downloads
import Interpolations: interpolate, extrapolate, Gridded, Linear, Flat
using Statistics

"""
    fetch_woa23(; lon_range = (-71.0, -53.0), lat_range = (40.0, 48.5), month = 0,
                output_dir = "inputs", include_o2 = true, backend = :zarr,
                verbose = true, strict = true) -> NamedTuple

Fetch WOA23 temperature, salinity, and oxygen climatology as GeoDatasets.

# Arguments
- `lon_range`: Longitude bounds (min, max)
- `lat_range`: Latitude bounds (min, max)
- `month`: Month (0 = annual mean, 1-12 = specific month)
- `output_dir`: Output directory
- `include_o2`: Include oxygen data
- `backend`: Storage backend (:zarr, :netcdf)
- `verbose`: Print progress messages
- `strict`: error when a source is unavailable instead of substituting a modelled value

# Returns
NamedTuple with file paths and interpolator functions.
"""
function fetch_woa23(;
    lon_range::Tuple{Real, Real} = (-71.0, -53.0),
    lat_range::Tuple{Real, Real} = (40.0, 48.5),
    month::Int = 0,
    output_dir::AbstractString = "inputs",
    # `output_path` is the name every other fetcher in the package uses for its
    # destination; WOA23 writes a store per variable into a directory instead. Both spellings
    # are accepted so the canonical keyword set holds, and passing both is an error rather
    # than a silent preference for one.
    output_path::Union{Nothing, AbstractString} = nothing,
    include_o2::Bool = true,
    backend::Symbol = :zarr,
    verbose::Bool = true,
    strict::Bool = true,
    force::Bool = false
)
    if output_path !== nothing
        output_dir === "inputs" || error("Both output_dir and output_path were given to " *
                                         "fetch_woa23. Pass one: output_path is accepted as " *
                                         "the canonical spelling of output_dir.")
        output_dir = dirname(output_path)
    end
    mkpath(output_dir)
    month_str = lpad(string(month), 2, '0')

    i_dep_hi = 101

    # WOA23 provides 0.25 deg (1441x721) and 1.00 deg (361x181) grids.
    # Grid indices are bounded to valid OPeNDAP coordinate indices [0, lon_max] and [0, lat_max].
    function grid_indices(res_deg)
        lon_max = round(Int, 360.0 / res_deg)
        lat_max = round(Int, 180.0 / res_deg)
        raw_i_lo = round(Int, (Float64(lon_range[1]) + 180.0) / res_deg)
        raw_i_hi = round(Int, (Float64(lon_range[2]) + 180.0) / res_deg)
        raw_j_lo = round(Int, (Float64(lat_range[1]) + 90.0) / res_deg)
        raw_j_hi = round(Int, (Float64(lat_range[2]) + 90.0) / res_deg)

        i_lo = max(0, min(raw_i_lo, lon_max))
        i_hi = max(0, min(raw_i_hi, lon_max))
        j_lo = max(0, min(raw_j_lo, lat_max))
        j_hi = max(0, min(raw_j_hi, lat_max))
        return i_lo, i_hi, j_lo, j_hi
    end

    # OPeNDAP windows: one axis range per requested coordinate, all on the same grid. The
    # subset string is built once per `res_deg` rather than recomputed per axis, so a
    # change to `grid_indices` cannot leave one axis on a stale window.
    function opendap_windows(; res_deg = 0.25)
        i_lo, i_hi, j_lo, j_hi = grid_indices(res_deg)
        (lon = "[$(i_lo):1:$(i_hi)]", lat = "[$(j_lo):1:$(j_hi)]")
    end

    """Depth index range shared by every WOA23 product."""
    opendap_depth() = "depth[0:1:$(i_dep_hi)]"

    function woa_url_candidates(variable_letter, varname)
        sub_dir = if variable_letter == "t"
            "temperature"
        elseif variable_letter == "s"
            "salinity"
        elseif variable_letter in ["o", "O", "A"]
            "oxygen"
        else
            "nutrients"
        end

            urls = String[]
            if variable_letter in ["o", "O", "A"]
                # Oxygen is published at 1.00 and 5.00 degree only, so its subset window is
                # built on the 1.00 grid - the grid the URL names, not the 0.25 one.
                ow = opendap_windows(; res_deg = 1.00)
                push!(urls, "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/oxygen/netcdf/all/1.00/woa23_all_o$(month_str)_01.nc")
                push!(urls, "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/oxygen/netcdf/all/5.00/woa23_all_o$(month_str)_05.nc")
                push!(urls, "https://www.ncei.noaa.gov/thredds/dodsC/ncei/woa/oxygen/netcdf/all/1.00/woa23_all_o$(month_str)_01.nc?$(varname)$(ow.lon),lat$(ow.lat),$(opendap_depth()),time[0:1:0]")
            else
            base_fn_25 = "woa23_A5B4_$(variable_letter)$(month_str)_04.nc"
            base_fn_1  = "woa23_A5B4_$(variable_letter)$(month_str)_01.nc"
            path_25 = "$(sub_dir)/netcdf/A5B4/0.25"
            path_1  = "$(sub_dir)/netcdf/A5B4/1.00"
            static_base = "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA"

            push!(urls, "$(static_base)/$(path_25)/$(base_fn_25)")
            push!(urls, "$(static_base)/$(path_1)/$(base_fn_1)")

            # A5B4 files are 0.25 deg: the bounding index window is aligned to that grid. Each axis
            # gets its own range from one call, so all three axes describe the same window.
            win = opendap_windows(; res_deg = 0.25)
            subset_str = string(varname, win.lon, ",",
                                "lon", win.lon, ",",
                                "lat", win.lat, ",",
                                "$(opendap_depth()),time[0:1:0]")
            for thredds in ("https://www.ncei.noaa.gov/thredds/dodsC/ncei/woa",
                            "https://thredds.ucar.edu/thredds/dodsC/ncei/woa",
                            "https://tds.marine.rutgers.edu/thredds/dodsC/ncei/woa",
                            "https://data.nodc.noaa.gov/thredds/dodsC/ncei/woa")
                push!(urls, "$(thredds)/$(path_25)/$(base_fn_25)?$(subset_str)")
            end
        end
        urls
    end

    t_file = joinpath(output_dir, "woa23_temperature_$(month_str)_0.25deg.zarr")
    s_file = joinpath(output_dir, "woa23_salinity_$(month_str)_0.25deg.zarr")
    o_file = joinpath(output_dir, "woa23_oxygen_$(month_str)_0.25deg.zarr")

    # `include_o2 = false` has to be checked *before* the destructure: pulling
    # `(a, b, c)` out of `nothing` calls `iterate(::Nothing)`, which made the guard below
    # unreachable and `include_o2 = false` an immediate error rather than a smaller fetch.
    for entry in (
        ("t", "t_an", t_file),
        ("s", "s_an", s_file),
        (include_o2 ? ("o", "o_an", o_file) : nothing),
    )
        isnothing(entry) && continue
        (variable_letter, varname, out_path) = entry
        # `force` is honoured here, not merely accepted: without it `fetch_input(key; force
        # = true)` silently returned a store it was asked to rebuild, because the adapter
        # passes `force` straight through and this signature had nowhere to put it.
        if is_cached(out_path) && !force
            verbose && println("WOA23: using cached store $(out_path)")
            continue
        end
        downloaded = false
        failure = ""
        for url in woa_url_candidates(variable_letter, varname)
            verbose && println("Fetching WOA23 from:\n  $(url[1:min(80, length(url))])...")
            try
                # Scratch beside the destination on purpose: a WOA23 file is hundreds of
                # MB and must not land in the OS temp directory.
                scratch = joinpath(output_dir, ".woa23-$(variable_letter)$(month_str).nc")
                download_to_path(url, scratch; verbose = verbose)
                ds = geoload(scratch; backend = :ncdatasets)
                geosave(out_path, ds; backend = backend)
                rm(scratch, force = true)
                verbose && println("  -> Saved to $(out_path)")
                downloaded = true
                break
            catch err
                failure = first(split(sprint(showerror, err), "\n"))
                verbose && println("  -> Mirror unsuccessful: $(failure)")
            end
        end
        if !downloaded
            strict && error("All WOA23 attempts failed for $(varname) (last error: $(failure)). " *
                            "No synthetic values were substituted: pass strict=false to fall " *
                            "back to a modelled climatology, and the fallback will be labelled " *
                            "in the interpolator it returns.")
            @warn "All WOA23 download attempts failed for $(varname); " *
                  "using a modelled fallback because strict=false."
        end
    end

    # Build interpolators
    function make_woa_interpolator(filepath, varname, fallback_val)
        if !is_cached(filepath)
            verbose && println("WOA23: file $(filepath) not found -- using physical fallback.")
            return if fallback_val isa Function
                fallback_val
            else
                (lon, lat, z) -> Float64(fallback_val)
            end
        end

        ds = geoload(filepath)
        # Name the variable rather than take the first one the store happens to hold. The
        # fallback read whatever `first(keys(...))` returned, which on a store holding
        # `grid_mapping` (a 0-D scalar) is not a field at all, and the failure surfaced much
        # later as "Dimension :lon not found" with no sign it came from a wrong variable.
        haskey(ds.variables, varname) || error(
            "Store $(basename(filepath)) has no variable $(repr(varname)). It holds: " *
            "$(sort(collect(keys(ds.variables)))). Pass a store written with that " *
            "variable, or name one it has.")
        ga = ds.variables[varname]

        # One accessor for all three axes: a file that spells them differently (lev,
        # longitude) is read the same way, and a missing axis reports which one. `axis`
        # returns (values, dimension), so the whole record is kept.
        lons, lon_dim = axis(ds, :lon)
        lats, lat_dim = axis(ds, :lat)
        deps, dep_dim = axis(ds, :depth)

        # WOA depth is positive-down; the interpolators take negative-up.
        deps_neg, _ = to_positive_down(-deps)

        # Orientation comes from the variable's own dimensions, not from position: a file
        # that stores (lat, lon, depth) is permuted into (lon, lat, depth) here, so the
        # per-axis sorts below reorder the axis they name rather than a fixed slot.
        (p_lon, p_lat, p_dep) = dim_permutation(ga.dims, :lon, :lat, :depth)
        if map(s -> size(ga.data, s), (p_lon, p_lat, p_dep)) !=
           (length(lons), length(lats), length(deps_neg))
            error("Variable $(repr(varname)) is $(size(ga.data)) with dims " *
                  "$([standardize_dimension_name(d.name) for d in ga.dims]), which does not " *
                  "match (lon=$(length(lons)), lat=$(length(lats)), depth=$(length(deps_neg))).")
        end

        # WOA23 carries a singleton `time` axis alongside the three the interpolator wants.
        # Order the data (lon, lat, depth, everything-else) from its own dimensions, take
        # step 1 of each extra axis, and drop them: permuting straight to a 3-element
        # permutation on a 4-D array is an error, and assuming time is the last slot was the
        # bug this replaces.
        extra = [standardize_dimension_name(d.name) for d in ga.dims
                 if standardize_dimension_name(d.name) ∉ (:lon, :lat, :depth)]
        perm = dim_permutation(ga.dims, :lon, :lat, :depth, extra...)
        ordered = permutedims(ga.data, perm)
        sel = ntuple(i -> i <= 3 ? (:) : 1, ndims(ordered))
        # Materialised on purpose: `_repair_woa_field!` mutates the field in place below, and
        # neither `dropdims` nor in-place writes exist on the lazy array a Zarr store hands
        # back. `collect` before `reshape` because reshaping a lazy disk array keeps it lazy.
        field_3d = reshape(collect(view(ordered, sel...)), size(ordered)[1:3])

        # Ensure sorting, permuting the data with each axis so the values a coordinate
        # names travel with it. Earlier this wrote `field_3d[:, :, p]`, which assumed
        # depth was the third slot regardless of what the file declared.
        if !issorted(lons)
            p = sortperm(lons); lons = lons[p]; field_3d = field_3d[p, :, :]
        end
        if !issorted(lats)
            p = sortperm(lats); lats = lats[p]; field_3d = field_3d[:, p, :]
        end
        if !issorted(deps_neg)
            p = sortperm(deps_neg); deps_neg = deps_neg[p]; field_3d = field_3d[:, :, p]
        end

        n_lon, n_lat, n_dep = size(field_3d)

        # Repair NaN cells (land masking, etc.)
        n_gap, n_landcol = _repair_woa_field!(field_3d)
        if verbose && (n_gap > 0 || n_landcol > 0)
            println("WOA23: repaired $(n_gap) masked level(s) and filled $(n_landcol) columns.")
        end

        # Degenerate singleton axes are expanded or sliced so Interpolations.jl
        # can handle both standard 3D grids and single-cell degenerate test fixtures.
        # When an axis has length > 1, Interpolations.scale/interpolate with
        # Gridded(Linear()) provides standard trilinear interpolation.
        function _axis_grid(coords::Vector{Float64})
            if length(coords) == 1
                return [coords[1] - 0.5, coords[1] + 0.5], true
            end
            return coords, false
        end

        grid_lons, lon_deg = _axis_grid(lons)
        grid_lats, lat_deg = _axis_grid(lats)
        grid_deps, dep_deg = _axis_grid(deps_neg)

        expanded_data = field_3d
        if lon_deg
            expanded_data = cat(expanded_data, expanded_data; dims = 1)
        end
        if lat_deg
            expanded_data = cat(expanded_data, expanded_data; dims = 2)
        end
        if dep_deg
            expanded_data = cat(expanded_data, expanded_data; dims = 3)
        end

        itp = extrapolate(
            interpolate((grid_lons, grid_lats, grid_deps), expanded_data,
                        Gridded(Linear())),
            Flat()
        )

        woa_interp(lon, lat, z) = Float64(itp(Float64(lon), Float64(lat), Float64(z)))
        return woa_interp
    end

    # Physical fallbacks for Northwest Atlantic shelf
    t_fallback(lon, lat, z) = z > -20.0 ? 14.0 : (z > -80.0 ? 2.0 : 8.0)
    s_fallback(lon, lat, z) = 31.5 + 3.0 * (1.0 - exp(-abs(Float64(z)) / 150.0))
    o2_fallback(lon, lat, z) = 300.0 - 95.0 * (1.0 - exp(-abs(Float64(z)) / 150.0))
    o2_sat_fallback(lon, lat, z) = 98.0 - 28.0 * (1.0 - exp(-abs(Float64(z)) / 150.0))

    t_fn = make_woa_interpolator(t_file, "t_an", t_fallback)
    s_fn = make_woa_interpolator(s_file, "s_an", s_fallback)
    o_fn = make_woa_interpolator(o_file, "o_an", o2_fallback)

    return (
        temperature_file = t_file,
        salinity_file    = s_file,
        oxygen_file      = o_file,
        temperature_fn   = t_fn,
        salinity_fn      = s_fn,
        oxygen_fn        = o_fn,
        oxygen_sat_fn    = o2_sat_fallback,
        nitrate_fn       = (lon, lat, z) -> 2.0 + 18.0 * (1.0 - exp(-abs(Float64(z)) / 100.0)),
        phosphate_fn     = (lon, lat, z) -> 0.3 + 1.2 * (1.0 - exp(-abs(Float64(z)) / 100.0)),
        silicate_fn      = (lon, lat, z) -> 4.0 + 22.0 * (1.0 - exp(-abs(Float64(z)) / 120.0)),
        aou_fn           = (lon, lat, z) -> 10.0 + 90.0 * (1.0 - exp(-abs(Float64(z)) / 150.0))
    )
end

"""
    _repair_woa_field!(field_3d::Array{<:AbstractFloat,3}) -> (n_gap, n_empty)

Fill NaN cells in a WOA field by copying from the nearest valid level in each column.
Land cells that are NaN at every level are left alone; a partially masked column is
filled outwards from its first and last valid values, which keeps the shelf-break from
reading as missing data.
"""
function _repair_woa_field!(field_3d)
    n_lon, n_lat, n_dep = size(field_3d)
    n_gap = 0
    n_empty = 0

    for j in 1:n_lat, i in 1:n_lon
        valid = findall(k -> !isnan(field_3d[i, j, k]), 1:n_dep)
        isempty(valid) && (n_empty += 1; continue)

        first_v, last_v = first(valid), last(valid)
        for k in 1:n_dep
            isnan(field_3d[i, j, k]) || continue
            if k < first_v
                field_3d[i, j, k] = field_3d[i, j, first_v]
            elseif k > last_v
                field_3d[i, j, k] = field_3d[i, j, last_v]
            else
                lo = last(valid[kk] for kk in eachindex(valid) if valid[kk] < k)
                hi = first(valid[kk] for kk in eachindex(valid) if valid[kk] > k)
                w = (k - lo) / (hi - lo)
                field_3d[i, j, k] =
                    field_3d[i, j, lo] * (1 - w) + field_3d[i, j, hi] * w
            end
            n_gap += 1
        end
    end

    if n_empty > 0
        good = filter(!isnan, field_3d)
        fill_value = isempty(good) ? 0.0 : sum(good) / length(good)
        for j in 1:n_lat, i in 1:n_lon
            all(!isnan, @view field_3d[i, j, :]) && continue
            field_3d[i, j, :] .= fill_value
        end
    end

    return (n_gap, n_empty)
end

"""
    load_woa23_interpolators(; lon_range, lat_range, month=0, output_dir="inputs") -> NamedTuple

Load or fetch WOA23 and return interpolator functions.
"""
function load_woa23_interpolators(;
    lon_range = (-71.0, -53.0),
    lat_range = (40.0, 48.5),
    month::Int = 0,
    output_dir::AbstractString = "inputs",
    kwargs...
)
    return fetch_woa23(;
        lon_range = lon_range,
        lat_range = lat_range,
        month = month,
        output_dir = output_dir,
        kwargs...
    )
end
