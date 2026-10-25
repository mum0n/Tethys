"""
High-level API: saving datasets.
"""

export geosave, save_dataset

"""
    save_dataset(uri::AbstractString, data::GeoDataset; backend=nothing, overwrite=false, kwargs...) -> GeoDataset

Save a GeoDataset to a URI, auto-detecting the backend from the file extension or scheme
when `backend` is not given.

`overwrite=false` updates a destination that is already there, for the backends that can:
NCDatasets opens the existing file in append mode, and a variable it already finds is
replaced in place. Zarr cannot — Zarr.jl reopens an existing store read-only and refuses to
create an array in a reopened group — so a second `save_dataset` to the same Zarr path is an error
naming `overwrite=true` rather than a half-written store. Pass `overwrite=true` to replace
the destination in both cases.

# Arguments
- `uri`: Destination path or URI
- `data`: GeoDataset to save
- `backend`: Backend name or instance
- `overwrite`: Replace an existing destination instead of updating it
- `kwargs`: Backend-specific options (`format`, ...)

# Examples
```julia
save_dataset("output/temperature.zarr", ds)
save_dataset("output/temperature.nc", ds; backend=:ncdatasets)
save_dataset("output/temperature.zarr", ds; overwrite=true)
```
"""
function save_dataset(uri::AbstractString, data::GeoDataset;
                      backend = nothing, overwrite::Bool = false, kwargs...)
    be = resolve_backend(backend, uri)
    path = strip_scheme(uri)
    backend_write(be, path, data; overwrite = overwrite, kwargs...)
    return data
end

const geosave = save_dataset

"""
    _save_with_backend(be, uri, data; kwargs...)

Escape hatch for callers holding a backend already; `geosave` resolves one for you.
"""
_save_with_backend(be::GeoBackend, uri::AbstractString, data::GeoDataset; kwargs...) =
    backend_write(be, strip_scheme(uri), data; kwargs...)
