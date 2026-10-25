"""
High-level API: loading datasets.
"""

export geoload, load_dataset, geoget, register_source_provider!, unregister_source_provider!,
       list_source_providers, open_dataset, close_dataset, infer_backend

# Global registry of source acquisition providers (e.g. :erddap, :copernicus, :open_meteo, :woa23)
const _SOURCE_PROVIDERS = Dict{Symbol, Function}()

"""
    register_source_provider!(scheme::Union{Symbol, AbstractString}, handler::Function)

Register an acquisition driver plugin for `geoget`. `handler` must accept
`(uri_or_urn::AbstractString; cache_dir, bounds, kwargs...)` and return a `GeoDataset`.
"""
function register_source_provider!(scheme::Union{Symbol, AbstractString}, handler::Function)
    _SOURCE_PROVIDERS[Symbol(scheme)] = handler
    return handler
end

"""
    unregister_source_provider!(scheme::Union{Symbol, AbstractString})

Unregister an acquisition provider plugin.
"""
function unregister_source_provider!(scheme::Union{Symbol, AbstractString})
    delete!(_SOURCE_PROVIDERS, Symbol(scheme))
    return nothing
end

"""
    list_source_providers() -> Vector{Symbol}

List registered source acquisition providers.
"""
list_source_providers() = sort!(collect(keys(_SOURCE_PROVIDERS)))

"""
    load_dataset(uri::String; backend::Union{Symbol, GeoBackend, Nothing}=nothing, kwargs...) -> GeoDataset

Load a dataset from a URI, auto-detecting the backend from the file extension or scheme.

# Arguments
- `uri`: Source URI (file path, `zarr://`, `netcdf://`, `nczarr://`, `geoparquet://`)
- `backend`: Backend name (`:zarr`, `:ncdatasets`, `:nczarr`, `:geoparquet`) or instance
- `kwargs`: Backend-specific options
"""
function load_dataset(uri::String; backend::Union{Symbol, GeoBackend, Nothing}=nothing, kwargs...)
    # If the file/dir does not exist locally and has a registered provider or URN scheme,
    # delegate to geoget to fetch and load it.
    if !isfile(uri) && !isdir(uri)
        m = match(r"^([a-zA-Z0-9_\-]+)://(.*)$", uri)
        provider_key = m !== nothing ? Symbol(lowercase(m[1])) : Symbol(lowercase(uri))
        if haskey(_SOURCE_PROVIDERS, provider_key)
            return geoget(uri; backend=backend, kwargs...)
        end
    end
    be = resolve_backend(backend, uri)
    return backend_open(be, uri; kwargs...)
end

const geoload = load_dataset
const open_dataset = load_dataset

"""
    geoget(
        uri_or_urn::AbstractString;
        cache_dir::Union{Nothing, AbstractString} = nothing,
        bounds = nothing,
        force::Bool = false,
        backend::Union{Symbol, GeoBackend, Nothing} = nothing,
        kwargs...
    ) -> GeoDataset

Unified acquisition and loading interface for `Tethys`.

If `uri_or_urn` points to an existing file/directory or local URI, it is loaded directly
via `geoload`. If it represents a registered catalog URN or provider URI (e.g.,
`erddap://...`, `copernicus://...`, `catalog://...` or a provider symbol key), `geoget`
dispatches to the registered provider driver plugin to acquire and return the `GeoDataset`.

# Examples
```julia
# Direct load
ds = geoget("inputs/bathymetry.nc")

# Provider acquisition via registered plugin
ds = geoget("erddap://ETOPO_2022_v1_15s"; bounds = ((-71.0, -53.0), (40.0, 48.5)))
```
"""
function geoget(
    uri_or_urn::AbstractString;
    cache_dir::Union{Nothing, AbstractString} = nothing,
    bounds = nothing,
    force::Bool = false,
    backend::Union{Symbol, GeoBackend, Nothing} = nothing,
    kwargs...
)
    # 1. If it exists on local filesystem or is an existing path, load directly
    if !force && (isfile(uri_or_urn) || isdir(uri_or_urn))
        return geoload(uri_or_urn; backend = backend, kwargs...)
    end

    # 2. Parse scheme / prefix if present (e.g. "erddap://...", "copernicus://...")
    m = match(r"^([a-zA-Z0-9_\-]+)://(.*)$", uri_or_urn)
    provider_key = m !== nothing ? Symbol(lowercase(m[1])) : Symbol(lowercase(uri_or_urn))

    if haskey(_SOURCE_PROVIDERS, provider_key)
        handler = _SOURCE_PROVIDERS[provider_key]
        return handler(
            uri_or_urn;
            cache_dir = cache_dir,
            bounds = bounds,
            force = force,
            backend = backend,
            kwargs...
        )
    end

    # 3. Fall back to geoload if no provider registered
    return geoload(uri_or_urn; backend = backend, kwargs...)
end

"""
    close_dataset(dataset::GeoDataset) -> Nothing

Close a dataset and release resources.
"""
function close_dataset(dataset::GeoDataset)
    if dataset.backend !== nothing
        backend_close(dataset.backend, dataset)
    end
    return nothing
end