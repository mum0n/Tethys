"""
Backend registry for Tethys.
"""
module StorageRegistry

using ..Types

const BACKEND_REGISTRY = Dict{Symbol, Any}()

"""
    register_backend(name::Symbol, backend)

Register a backend under a name.
"""
function register_backend(name::Symbol, backend)
    BACKEND_REGISTRY[name] = backend
    return backend
end

"""
    get_backend(name::Symbol) -> GeoBackend

Get a registered backend by name.
"""
function get_backend(name::Symbol)
    haskey(BACKEND_REGISTRY, name) || error("Unknown backend '$name'. Available: $(sort(collect(keys(BACKEND_REGISTRY))))")
    return BACKEND_REGISTRY[name]
end

"""
    list_backends() -> Vector{Symbol}

List all registered backend names.
"""
function list_backends()
    return sort(collect(keys(BACKEND_REGISTRY)))
end

"""
    infer_backend(uri::AbstractString) -> Symbol

Infer the backend from a URI scheme or file extension.

Throws when the URI does not identify a backend: a silent default hides a typo in a path.
"""
function infer_backend(uri::AbstractString)
    clean_p = replace(string(uri), r"^[a-zA-Z0-9]+://" => "")
    if isdir(clean_p) && (isfile(joinpath(clean_p, ".zgroup")) || isfile(joinpath(clean_p, ".zattrs")) || isfile(joinpath(clean_p, ".zarray")))
        return :zarr
    end
    u = lowercase(string(uri))
    startswith(u, "nczarr://") && return :nczarr
    endswith(u, ".zarr") && return :zarr
    startswith(u, "zarr://") && return :zarr
    occursin(".zarr/", u) && return :zarr
    endswith(u, ".nc") && return :ncdatasets
    startswith(u, "netcdf://") && return :ncdatasets
    endswith(u, ".parquet") && return :geoparquet
    startswith(u, "geoparquet://") && return :geoparquet
    error("Cannot infer a backend from URI '$(uri)'. Expected .zarr, .nc, .nczarr or " *
          ".parquet (or zarr://, netcdf://, nczarr://, geoparquet://), or pass an explicit " *
          "backend= keyword.")
end

export register_backend, get_backend, list_backends, infer_backend

end # module GeoDataRegistry