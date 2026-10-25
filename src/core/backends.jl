"""
    backends.jl

Unified storage backend interface for Tethys.
"""
module Backends

using ..Types: GeoDataset, GeoBackend, BackendCapabilities, Dimension

export GeoBackendInterface,
       backend_open, backend_close, backend_create, backend_write, backend_capabilities

"""
    backend_open(backend::GeoBackend, uri::AbstractString; kwargs...) -> GeoDataset

Open a persistent storage target and return an initialized `GeoDataset`.
"""
function backend_open end

"""
    backend_create(backend::GeoBackend, uri::AbstractString, dims::Dict{Symbol, Dimension}; kwargs...)

Create an empty persistent storage destination for `dims`.
"""
function backend_create end

"""
    backend_write(backend::GeoBackend, uri::AbstractString, dataset::GeoDataset; kwargs...)

Persist `dataset` into `uri` using `backend`.
"""
function backend_write end

"""
    backend_close(backend::GeoBackend, ds::GeoDataset; kwargs...)

Flush and release file handles associated with `ds`.
"""
function backend_close end

"""
    backend_capabilities(backend::GeoBackend) -> BackendCapabilities

Report supported features for `backend`.
"""
backend_capabilities(::GeoBackend) = BackendCapabilities(read = true, write = true)

const SUPPORTED_URI_SCHEMES = (
    "zarr://",
    "nczarr://",
    "netcdf://",
    "geoparquet://",
    "file://"
)

"""
    strip_scheme(uri::AbstractString) -> String

Remove a recognised `scheme://` prefix from `uri`, returning the local path portion.
"""
function strip_scheme(uri::AbstractString)
    s = string(uri)
    lower = lowercase(s)
    for prefix in SUPPORTED_URI_SCHEMES
        if startswith(lower, prefix)
            return s[length(prefix) + 1 : end]
        end
    end
    return s
end

export strip_scheme, SUPPORTED_URI_SCHEMES

end # module Backends
