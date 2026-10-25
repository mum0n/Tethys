# ==============================================================================
# boundary.jl -- Boundary hydrography fetch and interpolator construction
# Supports WOA23, GLORYS12v1 (Copernicus), HYCOM for sponge boundary conditions.
# ==============================================================================

using NCDatasets
import Interpolations: interpolate, extrapolate, Gridded, Linear, Flat
using Statistics

"""
    _central_glorys_subset(lon, lat; dir = nothing) -> Union{String, Nothing}

The GLORYS12V1 boundary subset already assimilated in the central store, when one covers the
requested box, else `nothing`. `dir` defaults to the `glorys` directory of the central store
and is a keyword so a test can pass a fixture directory without redirecting the store.

The store is searched because the per-domain cache path in `input_dir` is not where the
archive is kept: a subset downloaded once sits under the source's own directory, and looking
for it only by its per-domain name re-downloads what is on disk. Coverage is checked rather
than assumed, because a subset cut to a smaller box returns an empty sponge instead of an
error, which is the failure that looks like a successful run.

Latitude and longitude axes are read by name in either spelling — `longitude`/`latitude`, the
Copernicus convention — so a NetCDF using the other one is still measured rather than skipped.
"""
function _central_glorys_subset(lon::Tuple{Float64, Float64}, lat::Tuple{Float64, Float64};
                                dir::AbstractString = joinpath(store_root(), "glorys"))
    isdir(dir) || return nothing
    for name in sort(readdir(dir))
        f = joinpath(dir, name)
        endswith(name, ".nc") && isfile(f) || continue
        got_lon = got_lat = nothing
        try
            ds = NCDataset(f)
            try
                for ax in ("longitude", "lon")
                    if haskey(ds, ax)
                        v = ds[ax]
                        got_lon = (Float64(minimum(v)), Float64(maximum(v)))
                        break
                    end
                end
                for ax in ("latitude", "lat")
                    if haskey(ds, ax)
                        v = ds[ax]
                        got_lat = (Float64(minimum(v)), Float64(maximum(v)))
                        break
                    end
                end
            finally
                close(ds)
            end
        catch err
            # An unreadable NetCDF is not a candidate; the loop continues to the next one.
            continue
        end
        (got_lon === nothing || got_lat === nothing) && continue
        got_lon[1] <= lon[1] && got_lon[2] >= lon[2] &&
            got_lat[1] <= lat[1] && got_lat[2] >= lat[2] || continue
        return f
    end
    return nothing
end

"""
    fetch_boundary_hydrography_geodata(source::Symbol;
        lon_range, lat_range, input_dir,
        month = 0, verbose = true, with_velocities = false) -> NamedTuple

Acquire and load boundary temperature and salinity as sponge targets using Tethys.

# Arguments
- `source`: Data source (:woa23, :glorys12v1, :glorys_climatology, :hycom)
- `lon_range`: Longitude bounds (min, max)
- `lat_range`: Latitude bounds (min, max)
- `input_dir`: Input/output directory
- `month`: Month (0 = annual mean)
- `verbose`: Print progress messages
- `with_velocities`: Also return `u` and `v` interpolators (GLORYS only)

# Returns
NamedTuple with interpolator functions (T, S), and with `u` and `v` as well when
`with_velocities` is set.
"""
function fetch_boundary_hydrography_geodata(source::Symbol;
    lon_range, lat_range, input_dir,
    month::Int = 0, verbose::Bool = true, with_velocities::Bool = false)
    mkpath(input_dir)
    lon = (Float64(lon_range[1]), Float64(lon_range[2]))
    lat = (Float64(lat_range[1]), Float64(lat_range[2]))

    if source === :woa23
        mm = lpad(string(month), 2, '0')
        tp = joinpath(input_dir, "woa23_temperature_$(mm)_0.25deg.zarr")
        sp = joinpath(input_dir, "woa23_salinity_$(mm)_0.25deg.zarr")
        # `isdir || isfile`, because the product the fetcher writes *is* a Zarr store: a
        # directory. Testing `isfile` alone made this path fetch on every call and then
        # reject its own download -- "WOA23 boundary needs ... in <dir>" with the very files
        # the fetch had just saved into that directory sitting there. Same test as the
        # GLORYS branch below, which already had it right.
        if !(isdir(tp) || isfile(tp)) || !(isdir(sp) || isfile(sp))
            verbose && println("Fetching WOA23 climatology (keyless, NOAA)...")
            fetch_woa23(lon_range = lon_range, lat_range = lat_range,
                        month = month, output_dir = input_dir, backend=:zarr)
        end
        ((isdir(tp) || isfile(tp)) && (isdir(sp) || isfile(sp))) || error(
            "WOA23 boundary needs $(basename(tp)) and $(basename(sp)) in $(input_dir)")
        return build_boundary_tracer_interpolators_geodata(tp, sp; T_name = "t_an", S_name = "s_an")
    end

    if source in (:glorys12v1, :glorys_climatology)
        # The velocity-bearing subset is a DIFFERENT FILE from the temperature-and-salinity
        # one, not an extra variable in it. The reason is that the existing cache is
        # already on disk with two variables in it, and a cache that is checked for the
        # variables it holds would have to be re-fetched and re-written to gain two more.
        # A second path costs one download of overlapping data and leaves every existing
        # cache valid.
        suffix = with_velocities ? "_uv" : ""
        p = joinpath(input_dir, "boundary_glorys_1993_$(lon[1])_$(lon[2])_$(lat[1])_$(lat[2])$(suffix).zarr")
        if !isdir(p) && !isfile(p)
            nc = replace(p, ".zarr" => ".nc")
            # The central store holds a temperature-and-salinity subset. It has no
            # velocities, so a boundary that needs them is fetched from Copernicus
            # rather than silently served a subset that cannot answer the question.
            nc = isfile(nc) ? nc : (with_velocities ? nothing : _central_glorys_subset(lon, lat))
            if nc === nothing
                verbose && println("Fetching GLORYS12V1 boundary subset (Copernicus Marine)...")
                Copernicus.fetch_copernicus_physics_subset(
                    lon_range = lon_range, lat_range = lat_range,
                    start_date = "1993-01-01", end_date = "1993-12-31",
                    output_path = replace(p, ".zarr" => ".nc"),
                    dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1M-m",
                    with_velocities = with_velocities
                )
                nc = replace(p, ".zarr" => ".nc")
            else
                # Already assimilated. Re-downloading a subset that is on disk is the whole
                # reason the Copernicus console script is a dependency rather than the way
                # this normally runs; the error it raises when the script is missing is
                # indistinguishable from "the data is gone", so it is avoided.
                verbose && println("Using the GLORYS12V1 subset already assimilated: $nc")
            end
            # Convert to zarr. Whichever file the science wants, the interpolator below
            # reads one path, so the converted store is the cache and the .nc is kept as is.
            ds = geoload(nc; backend = :ncdatasets)
            geosave(p, ds; backend = :zarr)
        end
        m = month == 0 ? collect(1:12) : [max(1, min(12, month))]
        tracers = build_boundary_tracer_interpolators_geodata(p, p; T_name = "thetao", S_name = "so",
                                                              months = m)
        with_velocities || return tracers
        velocities = build_boundary_velocity_interpolators_geodata(p; u_name = "uo", v_name = "vo",
                                                                  months = m)
        return merge(tracers, velocities)
    end

    if source === :hycom
        p = joinpath(input_dir, "boundary_hycom.zarr")
        if !isdir(p) && !isfile(p)
            error(
                "HYCOM GOFS OPeNDAP server (tds.hycom.org) is currently unreachable or returns " *
                "HTTP 400. Use source = :woa23 or source = :glorys12v1 instead, or place a " *
                "pre-downloaded 'boundary_hycom.zarr' in '$(input_dir)'."
            )
        end
        return build_boundary_tracer_interpolators_geodata(p, p; T_name = "water_temp",
                                                            S_name = "salinity", months = [1])
    end

    error(
        "Unknown boundary source \"$(source)\". Choose :woa23, :glorys12v1, :hycom, or :synthetic."
    )
end

"""
    _boundary_axes(ds) -> (lons, lats, deps)

The coordinate vectors of a boundary dataset, with depth in the model's convention:
negative down, in metres.

Depth is normalised here rather than at each use because the two boundary products
disagree about it: WOA23 stores depth positive down, GLORYS stores it positive up, and
an interpolator built on the wrong sign of `z` is not a wrong answer, it is an answer
about the wrong place.
"""
function _boundary_axes(ds)
    lons, _ = axis(ds, :lon)
    lats, _ = axis(ds, :lat)
    deps, _ = axis(ds, :depth)
    if !isempty(deps) && minimum(deps) > -0.5 * maximum(abs, deps)
        deps = -deps
    end
    return lons, lats, deps
end

"""
    _boundary_variable_data(ds, name, months) -> Array

One variable of a boundary dataset as a 3-D (lon, lat, depth) array, with the time axis
resolved to the requested months.
"""
function _boundary_variable_data(ds, name::AbstractString, months::Vector{Int})
    data = ds.variables[name].data
    if ndims(data) == 4 && length(months) > 1
        data = mean(data[:, :, :, months], dims = 4)[:, :, :, 1]
    elseif ndims(data) == 4
        data = data[:, :, :, months[1]]
    end
    lons, _ = axis(ds, :lon)
    lats, _ = axis(ds, :lat)
    if size(data, 1) == length(lats) && size(data, 2) == length(lons)
        data = permutedims(data, (2, 1, 3))
    end
    return data
end

"""
    _boundary_permutations(lons, lats, deps) -> (plon, plat, pdep)

The permutations that bring the coordinate vectors into ascending order.

They are computed once and applied to every variable, because a temperature sorted one
way and a velocity sorted another are two different fields on two different grids
wearing the same axes.
"""
function _boundary_permutations(lons, lats, deps)
    plon = issorted(lons) ? collect(eachindex(lons)) : sortperm(lons)
    plat = issorted(lats) ? collect(eachindex(lats)) : sortperm(lats)
    pdep = issorted(deps) ? collect(eachindex(deps)) : sortperm(deps)
    return plon, plat, pdep
end

"""
    _apply_boundary_permutation(data, plon, plat, pdep) -> data
"""
function _apply_boundary_permutation(data, plon, plat, pdep)
    return data[plon, :, :][:, plat, :][:, :, pdep]
end

"""
    _fill_boundary_nans!(data, deps) -> data

Fill NaNs downward through the water column with the deepest valid value above them.

The fill is downward because the boundary data is a reanalysis whose levels stop at the
sea floor: below the deepest wet level of a column there is no value, and the water
there is the water above it.
"""
function _fill_boundary_nans!(data, deps)
    for j in 1:size(data, 2), i in 1:size(data, 1)
        last_valid = NaN
        for k in length(deps):-1:1
            if !isnan(data[i, j, k])
                last_valid = data[i, j, k]
            elseif !isnan(last_valid)
                data[i, j, k] = last_valid
            end
        end
    end
    return data
end

"""
    _replace_boundary_nans!(data, default) -> data
"""
function _replace_boundary_nans!(data, default::Real)
    replace!(data, NaN => Float64(default))
    return data
end

"""
    _boundary_interpolator(lons, lats, deps, data) -> (lon, lat, z, t) -> value
"""
function _boundary_interpolator(lons, lats, deps, data)
    itp = interpolate((lons, lats, deps), data, Gridded(Linear()))
    itp_ext = extrapolate(itp, Flat())
    return (lon, lat, z, t = 0.0) -> Float64(itp_ext(Float64(lon), Float64(lat), Float64(z)))
end

"""
    build_boundary_tracer_interpolators_geodata(temp_file, salt_file; T_name, S_name, months=[1]) -> NamedTuple

Build (lon, lat, z, t) interpolators for boundary T and S from Tethys files.

# Arguments
- `temp_file`: Path to temperature GeoDataset
- `salt_file`: Path to salinity GeoDataset
- `T_name`: Temperature variable name
- `S_name`: Salinity variable name
- `months`: Months to average (if time dimension present)

# Returns
NamedTuple with (T, S) interpolator functions.
"""
function build_boundary_tracer_interpolators_geodata(
    temp_file::AbstractString, salt_file::AbstractString;
    T_name::AbstractString = "t_an", S_name::AbstractString = "s_an",
    months::Vector{Int} = [1]
)
    # Load temperature GeoDataset
    temp_ds = geoload(temp_file)
    salt_ds = geoload(salt_file)

    # Get coordinate arrays. `axis` returns (values, Dimension), so the dimension record is
    # dropped here: everything below works on the values, and taking the pair as-is made
    # `minimum(deps)` fail on a tuple of (Vector, Dimension) the first time this path ran.
    # `axis` names what is missing rather than silently handing back nothing, and it resolves
    # aliases, so `t` or `datetime` is found too.
    lons, lats, deps = _boundary_axes(temp_ds)

    # Time is optional: a static source legitimately has no time axis. The presence test
    # goes through the same alias table `axis` uses, so a file that spells the axis `t`
    # does not read as "no time axis" and silently fall back to a single frame.
    times = any(k -> standardize_dimension_name(k) === :time,
                keys(temp_ds.coords)) ? first(axis(temp_ds, :time)) : Float64[0.0]

    # Get data arrays
    T_data = _boundary_variable_data(temp_ds, T_name, months)
    S_data = _boundary_variable_data(salt_ds, S_name, months)

    # Sort coordinates, once, and apply the same permutation to every variable.
    plon, plat, pdep = _boundary_permutations(lons, lats, deps)
    lons, lats, deps = lons[plon], lats[plat], deps[pdep]
    T_data = _apply_boundary_permutation(T_data, plon, plat, pdep)
    S_data = _apply_boundary_permutation(S_data, plon, plat, pdep)

    # Fill downward from the deepest valid value, then replace what is left with the
    # domain average: a fully dry land column has no water in it at all, and the sponge
    # should not be asked about it.
    _fill_boundary_nans!(T_data, deps)
    _fill_boundary_nans!(S_data, deps)

    valid_T_all = filter(!isnan, T_data)
    valid_S_all = filter(!isnan, S_data)
    default_T = isempty(valid_T_all) ? 10.0 : mean(valid_T_all)
    default_S = isempty(valid_S_all) ? 35.0 : mean(valid_S_all)
    _replace_boundary_nans!(T_data, default_T)
    _replace_boundary_nans!(S_data, default_S)

    # Create 3D interpolators (lon, lat, depth)
    return (T = _boundary_interpolator(lons, lats, deps, T_data),
            S = _boundary_interpolator(lons, lats, deps, S_data))
end

"""
    build_boundary_velocity_interpolators_geodata(vel_file; u_name = "uo", v_name = "vo",
                                                  months = [1]) -> NamedTuple

Build (lon, lat, z, t) interpolators for the boundary zonal and meridional velocity.

This is the quantity the momentum sponge relaxes toward, and it is the one the
temperature-and-salinity boundary cannot supply: a sponge that fixes what the incoming
water *is* and not how fast it is *going* constrains a domain whose open edges are its
main water source on two hand-entered numbers.

The depth sign is the model's (negative down), and the value substituted for a column
with no valid velocity is ZERO rather than the domain mean. A mean flow substituted
for missing data is a current invented out of nothing, and the momentum sponge would
then drive the interior toward it; still water is at least an honest statement that
nothing is known there.

The interpolation is in time-independent form: the result is a monthly climatology, and
the tidal signal is supplied separately by the tidal forcing.
"""
function build_boundary_velocity_interpolators_geodata(
    vel_file::AbstractString;
    u_name::AbstractString = "uo", v_name::AbstractString = "vo",
    months::Vector{Int} = [1]
)
    ds = geoload(vel_file)
    lons, lats, deps = _boundary_axes(ds)

    u_data = _boundary_variable_data(ds, u_name, months)
    v_data = _boundary_variable_data(ds, v_name, months)

    plon, plat, pdep = _boundary_permutations(lons, lats, deps)
    lons, lats, deps = lons[plon], lats[plat], deps[pdep]
    u_data = _apply_boundary_permutation(u_data, plon, plat, pdep)
    v_data = _apply_boundary_permutation(v_data, plon, plat, pdep)

    _fill_boundary_nans!(u_data, deps)
    _fill_boundary_nans!(v_data, deps)
    _replace_boundary_nans!(u_data, 0.0)
    _replace_boundary_nans!(v_data, 0.0)

    return (u = _boundary_interpolator(lons, lats, deps, u_data),
            v = _boundary_interpolator(lons, lats, deps, v_data))
end
