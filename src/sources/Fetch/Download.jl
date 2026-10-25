# ==============================================================================
# Fetch/Download.jl -- Atomic resilient downloading with retries
# ==============================================================================

const DEFAULT_DOWNLOAD_RETRIES = 3
const DEFAULT_CONNECT_TIMEOUT = 600

"""
    download_to_path(url, path; retries=3, timeout=600, verbose=false, headers=(), expected_bytes=nothing) -> path

Download `url` to `path` atomically: write `path.part`, then move it into place.

- `retries` attempts, each waiting `min(2^attempt, 30)` seconds.
- `timeout` is the connect timeout in seconds.
- `headers` are passed through to `Downloads`.
- `expected_bytes`, when given, verifies the downloaded size and refuses an incomplete body.

The scratch file is a sibling of the destination, so nothing is written outside the
project or the OS temp directory, and a cancelled download leaves no `.part` behind.
"""
function download_to_path(url::AbstractString, path::AbstractString;
                          retries::Int = DEFAULT_DOWNLOAD_RETRIES,
                          timeout::Int = DEFAULT_CONNECT_TIMEOUT,
                          verbose::Bool = false,
                          headers = (),
                          expected_bytes = nothing)
    retries >= 1 || error("download_to_path needs at least one attempt, got $retries")
    mkpath(dirname(path))
    part = string(path, ".part")
    ispath(part) && rm(part; force = true)

    last_error = nothing
    for attempt in 1:retries
        try
            # A header collection has to be a concrete `Vector{Pair}`: `Downloads` rejects
            # a lazy iterable, and the tuple literal `()` is not a `Pair` iterable at all
            # once it passes through the generic call path. Copying into what the type
            # promises is cheap and keeps the empty case valid.
            Downloads.download(url, part; timeout = timeout,
                               headers = collect(Pair{String, String}, headers))
            if expected_bytes !== nothing && filesize(part) != Int(expected_bytes)
                error("expected $(expected_bytes) bytes, got $(filesize(part))")
            end
            mv(part, path; force = true)
            verbose && println("  downloaded $(basename(path)) ($(filesize(path)) bytes)")
            return path
        catch err
            last_error = err
            ispath(part) && rm(part; force = true)
            if attempt < retries
                wait = min(2.0^attempt, 30.0)
                verbose && println("  attempt $attempt failed; retrying in $(wait)s")
                sleep(wait)
            end
        end
    end
    error("Downloading $url failed after $(retries) attempt(s). Last error: " *
          "$(sprint(showerror, last_error))")
end

"""
    download_ranges_parallel(url, path; nstreams, chunk, retries, verbose) -> path

Fetch `url` to `path` using parallel HTTP range requests, each stream writing its own piece
at its own file offset.

This exists because one source needs it. The TPXO9-Atlas archive is 4.7 GB and Zenodo serves
it at ~0.7 MB/s on a single stream, which is ~106 minutes — long enough that the default
three-attempt retry budget in `download_to_path` can never cover it. Eight streams measured
~4.4 MB/s, about 17 minutes, on the same URL.

Nothing here interprets the archive. It is the same bytes, fetched faster; the ZIP is read
afterwards by `ZipFile.Reader` in `Fetch/tides.jl`. The file size comes from a HEAD request
first, so an unsized server is an error rather than a silent guess.

Whether the server honours byte ranges is *not* read off `Accept-Ranges`. That header is
optional, and the one source here proves it: Zenodo's `/content` endpoint serves the 4.7 GB
TPXO9 archive, answers a range request with `206` and exactly the requested span, and sends
no `Accept-Ranges` header at all — so requiring it refuses a download that works. Instead
every piece is checked against the span it asked for, and a response that is the wrong size
is an error naming both, rather than bytes written at the wrong offset.

Each attempt rewrites only its own piece, so an interruption costs the remaining chunks
rather than the whole file — which is the other thing `download_to_path` cannot do, since it
deletes its partial on each retry.
"""
function download_ranges_parallel(url::AbstractString, path::AbstractString;
                                  nstreams::Int = 8,
                                  chunk::Int = 16 * 1024 * 1024,
                                  retries::Int = 5,
                                  verbose::Bool = false)
    ispath(path) && return path

    head = HTTP.head(url; retry = false)
    total = something(tryparse(Int, HTTP.header(head, "Content-Length", "")), 0)
    total > 0 || error("$(url) returned no Content-Length, so its size is unknown and the " *
                      "download cannot be split by offset.")

    mkpath(dirname(path))
    part = string(path, ".part")
    open(part, "w+") do io
        truncate(io, total)                    # placeholders so each seek lands in the file
    end

    fetched = 0
    t0 = time()
    offset = 0
    while offset < total
        tasks = Task[]
        ranges = Tuple{Int, Int}[]
        i = 0
        while i < nstreams && offset + i * chunk < total
            a = offset + i * chunk
            b = min(a + chunk, total) - 1
            push!(ranges, (a, b))
            push!(tasks, @async begin
                attempt = 0
                while true
                    try
                        r = HTTP.get(url; headers = ["Range" => "bytes=$a-$b"],
                                     retry = false)
                        length(r.body) == b - a + 1 || error(
                            "asked for bytes $a-$b ($(b - a + 1)) but got $(length(r.body))")
                        open(part, "r+") do f
                            seek(f, a)
                            write(f, r.body)
                        end
                        return length(r.body)
                    catch e
                        attempt += 1
                        attempt >= retries &&
                            error("range $a-$b failed after $retries attempts: " *
                                  "$(first(split(sprint(showerror, e), "\n")))")
                        sleep(2.0 ^ attempt)
                    end
                end
            end)
            i += 1
        end
        for t in tasks
            fetched += fetch(t)
        end
        offset += i * chunk
        if verbose
            rate = fetched / 1024^2 / max(1e-6, time() - t0)
            println("  $(round(100 * offset / total, digits = 1))%  " *
                    "$(round(fetched / 1024^2, digits = 1)) / " *
                    "$(round(total / 1024^2, digits = 1)) MB  " *
                    "$(round(rate, digits = 2)) MB/s")
        end
    end
    close(open(part, "a"))                     # drop the write handle before renaming

    mv(part, path; force = true)
    filesize(path) == total || error("fetched $(filesize(path)) bytes but the server " *
                                    "declared $total.")
    verbose && println("  downloaded $(basename(path)) " *
                       "($(round(total / 1024^2, digits = 1)) MB) in " *
                       "$(round(time() - t0, digits = 1))s")
    return path
end


"""
    cache_path(key::Symbol, output_dir; ext::String = "", subdir::String = "inputs") -> String

Canonical cache location for a declared source key under `output_dir`.

One convention, shared by `Manifest.input_dir` and every fetcher: caches live in
`<output_dir>/<subdir>/` and are named after the key.
"""
function cache_path(key::Symbol, output_dir::AbstractString; ext::String = "",
                    subdir::String = "inputs")
    joinpath(output_dir, subdir, string(key, ext))
end

export DEFAULT_DOWNLOAD_RETRIES, DEFAULT_CONNECT_TIMEOUT, download_to_path, cache_path
