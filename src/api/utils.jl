"""
Shared utilities for the Tethys API.
"""

export resolve_backend

"""
    resolve_backend(backend, uri) -> GeoBackend

Resolve a backend specification to an instance: `nothing` infers the backend from the URI
(see `infer_backend`), a `Symbol` looks one up in the registry, an instance is used as
given.

This is the only place a backend name becomes an instance, and it is used by `geoload`
and `geosave` - the two operations that actually need a backend. The coordinate
operations do not: they work on any `GeoDataset` regardless of where it came from.
"""
function resolve_backend(backend, uri)
    backend === nothing && return get_backend(infer_backend(uri))
    backend isa Symbol && return get_backend(backend)
    return backend
end
