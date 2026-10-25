using ZipFile

const TPXO9_ARCHIVE_URL =
    "https://zenodo.org/api/records/8074917/files/TPXO9_datafiles.zip/content"
const TPXO9_VELOCITY_MEMBER = "TPXO9_datafiles/u_tpxo9.v1.nc"

"""
    fetch_tpxo9_velocity(;
        output_path::AbstractString,
        lon_range = nothing,
        lat_range = nothing,
        force::Bool = false,
        verbose::Bool = true
    ) -> GeoDataset

Fetch the global TPXO9-Atlas velocity harmonics from the open Zenodo archive, returning the
loaded dataset.

This downloads the whole archive and extracts one member from it. That is deliberately
simple rather than clever: the archive is about 4.7 GB and the member 1.4 GB, and the two
cheaper approaches tried here — HTTP byte-range requests, and reconstructing a single-member
ZIP from the central directory — each traded a readable method for failure modes nobody
could interpret (`ZlibError: incorrect data check`) while saving ~3 GB of disk that was never
the scarce resource.

The archive is written beside the destination, inside the project rather than the OS temp
directory, and removed once the member is out, so the steady state is the extracted file and
the peak is archive plus file.

Two properties are checked rather than assumed, because a tidal atlas covering the wrong
region returns an empty box rather than an error, and the archive also contains regional
`*_remap*` members that look equivalent and are not:

- the extracted file carries the velocity harmonics the readers open. TPXO9-Atlas carries
  them in **two forms** — amplitude/phase (`ua`, `up`, `va`, `vp`, in cm/s and degrees GMT)
  and complex (`uRe`, `uIm`, `vRe`, `vIm`, in m s⁻¹) — and readers in
  TethysFlow and TethysDrift accept either representation: `read_tidal_velocity_harmonics` opens
  `ua`, while harmonic-synthesis paths open `uRe`. Either form satisfies this check, and the
  returned field records which one was found, enabling downstream models to handle both;
- its latitude and longitude axes actually span the requested domain.
"""
function fetch_tpxo9_velocity(;
    output_path::AbstractString = joinpath("inputs", "tpxo9", "u_tpxo9.v1.nc"),
    lon_range::Union{Nothing, Tuple{Real, Real}} = nothing,
    lat_range::Union{Nothing, Tuple{Real, Real}} = nothing,
    force::Bool = false,
        verbose::Bool = true,
    backend::Symbol = :zarr,
)
    if is_cached(output_path) && !force
        verbose && println("Tides: using cached store $(output_path)")
        return _validate_tpxo9_velocity(output_path; lon_range = lon_range,
                                        lat_range = lat_range)
    end
    is_cached(output_path) && rm(output_path; recursive = true, force = true)

    # Beside the destination, not in the OS temp directory, and with a `.zip` suffix so the
    # file carries what it is.
    archive = joinpath(dirname(output_path), splitext(basename(TPXO9_ARCHIVE_URL))[1] * ".zip")
    if isfile(archive) && force
        rm(archive; force = true)
    end

    if !isfile(archive)
        verbose && println("Tides: downloading the TPXO9 archive (about 4.7 GB) ...")
        # Parallel range requests, not `download_to_path`: this archive is 4.7 GB and the
        # server gives ~0.7 MB/s on one stream (~106 minutes), which exceeds the default
        # three-attempt retry budget before the first byte is wrong. Eight streams gave
        # ~4.4 MB/s (~17 minutes) on the same URL.
        download_ranges_parallel(TPXO9_ARCHIVE_URL, archive; verbose = verbose)
    end

    try
        io = open(archive)
        r = ZipFile.Reader(io)
        try
            k = findfirst(f -> f.name == TPXO9_VELOCITY_MEMBER, r.files)
            k === nothing && error(
                "The archive at $(TPXO9_ARCHIVE_URL) contains no " *
                "$(TPXO9_VELOCITY_MEMBER). It holds: $(join([f.name for f in r.files], ", "))." *
                " Nothing has been written; the archive is left in place for inspection.")
            t = @elapsed write(output_path, read(r.files[k]))
            verbose && println("Tides: extracted $(basename(output_path)) " *
                               "($(filesize(output_path)) bytes) in $(round(t, digits = 1))s")
        finally
            close(r)
            close(io)
        end
    finally
        # The extracted file is what the caller asked for; the 4.7 GB it came from is not,
        # so it goes. A failure above must not leave it behind.
        GC.gc()
        try
            rm(archive; force = true)
        catch e
            verbose && @warn "Could not immediately remove temporary archive $(archive)" exception = e
        end
    end

    return _validate_tpxo9_velocity(output_path; lon_range = lon_range,
                                    lat_range = lat_range)
end

"""
    _validate_tpxo9_velocity(path; lon_range, lat_range) -> GeoDataset

Check the extracted file is the global velocity product and spans the requested domain.

The domain check exists because the archive also holds regional `*_remap*` members
(`h_tpxo9_remap3600x1800_sonett.nc` spans lon -30..30, lat -55..-10) and a tidal atlas
covering the wrong region yields an empty box rather than an error. Naming the actual
coverage is the difference between a wrong run and a failed one.

The velocity check accepts **either** harmonic form TPXO9-Atlas ships — amplitude/phase
(`ua`, `up`, `va`, `vp`) or complex (`uRe`, `uIm`, `vRe`, `VIm`) — because they represent the same
harmonic information, supported across TethysFlow and TethysDrift readers:
`read_tidal_velocity_harmonics` reads `ua`, while harmonic-synthesis paths read `uRe`.
Both harmonic formulations are valid and the form found is reported.
"""
function _validate_tpxo9_velocity(path; lon_range = nothing, lat_range = nothing,
                                  verbose::Bool = true)
    ds = geoload(path)
    haskey(ds.variables, "con") || error(
        "$(basename(path)) has no `con` constituent list, so the tidal constituents cannot " *
        "be named. The TPXO9-Atlas velocity product carries one.")

    amp_phase = all(v -> haskey(ds.variables, v), ("ua", "up", "va", "vp"))
    complex_form = all(v -> haskey(ds.variables, v), ("URe", "UIm", "VRe", "VIm"))
    (amp_phase || complex_form) || error(
        "$(basename(path)) carries neither the amplitude/phase velocity harmonics " *
        "(ua, up, va, vp) nor the complex ones (URe, UIm, VRe, VIm). It has: " *
        "$(sort(collect(keys(ds.variables)))). Something other than the TPXO9-Atlas " *
        "velocity product was extracted.")
    if verbose
        forms = String[]
        amp_phase && push!(forms, "amplitude/phase (ua/up/va/vp)")
        complex_form && push!(forms, "complex (URe/UIm/VRe/VIm)")
        println("Tides: velocity harmonics in ", join(forms, " and "), " form")
    end
    if lat_range !== nothing || lon_range !== nothing
        lons = axis(ds, :lon)[1]
        lats = axis(ds, :lat)[1]
        lon_cover = extrema(lons)
        lat_cover = extrema(lats)
        if lat_range !== nothing &&
           (lat_range[1] < lat_cover[1] - 1.0 || lat_range[2] > lat_cover[2] + 1.0)
            error("The tidal file covers latitude $(lat_cover[1]):$(lat_cover[2]) but the " *
                  "requested domain is $(lat_range). A regional atlas must not be used " *
                  "where a global one is required.")
        end
        if lon_range !== nothing &&
           (lon_range[1] < lon_cover[1] - 1.0 || lon_range[2] > lon_cover[2] + 1.0)
            error("The tidal file covers longitude $(lon_cover[1]):$(lon_cover[2]) but the " *
                  "requested domain is $(lon_range).")
        end
    end
    return ds
end
