"""
    zarr_backend.jl

Zarr v2 storage backend.

Zarr v2 stores arrays with no dimension metadata at all, so axes are inferred:
1-D arrays whose canonical name is a standard dimension name (`lon`, `lat`, `depth`,
`time`, `x`, `y`, `z`) become coordinate variables; every other array is a data variable.
That heuristic is wrong for an unlabelled 1-D time series, so a self-describing backend
(the NetCDF one) is preferable when the file format can be chosen.

`backend_open` returns the chunked Zarr array itself, so opening a store reads nothing
from disk; indexing then materialises only the elements it touches. Coordinates are read
in full because every selection is resolved by searching them.
"""

using Zarr

struct ZarrBackend <: GeoBackend end

"""
    _zarr_dims_to_tuple(var_shape, dims, coords, attrs=Dict()) -> Tuple{Vararg{Dimension}}

Dimension records for a variable of shape `var_shape`, matched against explicit dimension
attributes (`_ARRAY_DIMENSIONS`) if present, or inferred from `dims` and `coords`.
"""
function _zarr_dims_to_tuple(var_shape, dims::Dict{Symbol, Dimension},
                             coords::Dict{Symbol, GeoArray},
                             attrs::AbstractDict = Dict{String, Any}())
    dim_attr = get(attrs, "_ARRAY_DIMENSIONS", nothing)
    names = if dim_attr isa AbstractVector && length(dim_attr) == length(var_shape)
        [standardize_dimension_name(Symbol(string(d))) for d in dim_attr]
    else
        infer_dim_names_from_size(var_shape, dims)
    end
    refs = Dimension[]
    for i in 1:length(var_shape)
        name = names[i]
        coord_ga = get(coords, name, nothing)
        if coord_ga === nothing && !haskey(dims, name)
            # An axis no declared dimension matches: name it, values unknown.
            push!(refs, Dimension(name = name, size = var_shape[i], coords = nothing))
            continue
        end
        source = coord_ga === nothing ? dims[name] : coord_ga.dims[1]
        push!(refs, Dimension(name = name, size = var_shape[i],
                              coords = source.coords,
                              units = source.units,
                              standard_name = source.standard_name,
                              calendar = source.calendar))
    end
    return tuple(refs...)
end

function backend_open(backend::ZarrBackend, uri::AbstractString; kwargs...)
    path = strip_scheme(uri)
    isdir(path) || error("Cannot open Zarr store '$(uri)': directory does not exist.")
    mode = haskey(kwargs, :mode) ? kwargs[:mode] : "r"
    g = Zarr.zopen(path, mode)

    array_names = String[]
    for k in keys(g.arrays)
        obj = g.arrays[k]
        obj isa Zarr.ZArray || continue
        push!(array_names, k)
    end

    coord_names, data_names = String[], String[]
    for k in array_names
        canon = standardize_dimension_name(Symbol(k))
        if ndims(g.arrays[k]) == 1 && canon in (:lon, :lat, :depth, :time, :x, :y, :z)
            push!(coord_names, k)
        else
            push!(data_names, k)
        end
    end

    crs = CoordinateSystem(crs = something(get(g.attrs, "crs", nothing), "EPSG:4326"))
    coords = Dict{Symbol, GeoArray}()
    dims = Dict{Symbol, Dimension}()

    # A store that declared its dimensions carries them, and they win. The name heuristic
    # below can only recognise the geographical axes -- lon/lat/depth/time/x/y/z -- so a
    # store holding anything else (spatial units, ensemble draws, a month index) would have
    # its axes read back as data variables, and a reader would have to know by hand that
    # "auids" is the spatial axis. `backend_write` records the dimensions the dataset
    # declared, so the store describes itself.
    declared = _declared_dimensions(get(g.attrs, "dimensions", nothing))
    for (name, d) in declared
        dims[name] = d
        if d.coords !== nothing
            obj = get(g.arrays, string(name), nothing)
            c = obj === nothing ? nothing : collect(Float64, vec(obj[:]))
            d = Dimension(name = d.name, size = d.size, coords = c, units = d.units,
                          standard_name = d.standard_name, dim_type = d.dim_type)
            dims[name] = d
            coords[name] = GeoArray(c, (d,), crs, Dict{String, Any}())
        end
    end

    for k in coord_names
        haskey(dims, standardize_dimension_name(Symbol(k))) && continue
        obj = g.arrays[k]
        canon = standardize_dimension_name(Symbol(k))
        c = collect(Float64, vec(obj[:]))
        d = Dimension(name = canon, size = length(c), coords = c,
                      units = get(obj.attrs, "units", nothing),
                      standard_name = get(obj.attrs, "standard_name", nothing),
                      calendar = get(obj.attrs, "calendar", nothing))
        dims[canon] = d
        coords[canon] = GeoArray(c, (d,), crs, Dict(obj.attrs))
    end

    vars = Dict{String, GeoArray}()
    for k in data_names
        haskey(dims, Symbol(k)) && continue
        obj = g.arrays[k]
        vars[k] = GeoArray(obj, _zarr_dims_to_tuple(size(obj), dims, coords, obj.attrs),
                           crs, Dict(obj.attrs))
    end

    return GeoDataset(vars, coords, dims, crs, Dict(g.attrs), ZarrBackend(), path)
end

"""
    _dimension_type(name) -> DimensionType

The dimension type stored under `name`, or `DIM_GENERIC` if it names none. `DimensionType`
is an enum, so only its name survives a JSON round trip.
"""
function _dimension_type(name)
    s = name === nothing ? "" : string(name)
    for t in instances(DimensionType)
        string(t) == s && return t
    end
    return DIM_GENERIC
end

"""
    _declared_dimensions(record) -> Vector{Pair{Symbol, Dimension}}

The dimension records a store wrote into its group attributes, as `Dimension`s, or empty.

The record is what `backend_write` wrote: a vector of dictionaries with `name`, `size`,
`units`, `standard_name` and `dim_type`, plus `coords` when the dimension had a coordinate
array. Anything that does not parse is dropped rather than guessed at — a dimension named
wrongly is a dimension a reader indexes wrongly, and the fallback heuristic covers the
geographical axes anyway.
"""
function _declared_dimensions(record)
    out = Pair{Symbol, Dimension}[]
    record isa AbstractVector || return out
    for entry in record
        entry isa AbstractDict || continue
        name = haskey(entry, "name") ? Symbol(entry["name"]) : nothing
        name === nothing && continue
        size_ = haskey(entry, "size") ? Int(entry["size"]) : 0
        size_ > 0 || continue
        coords = get(entry, "coords", nothing)
        push!(out, name => Dimension(name = name, size = size_,
                                     coords = coords === nothing ? nothing :
                                              collect(Float64, coords),
                                     units = get(entry, "units", nothing),
                                     standard_name = get(entry, "standard_name", nothing),
                                     # `dim_type` is an enum, and an enum does not survive
                                     # JSON as anything but its name. `DIM_` keeps the
                                     # reader's terminology, and a name that does not match
                                     # any of them is `DIM_GENERIC` rather than an error,
                                     # because a wrong dimension *type* degrades a search
                                     # and a wrong *name* indexes the wrong axis.
                                     dim_type = _dimension_type(get(entry, "dim_type", nothing)),
                                     calendar = get(entry, "calendar", nothing)))
    end
    return out
end

"""
    create_zarr_template(uri::AbstractString; dims, attrs, variables) -> Zarr.Group

Create a Zarr store whose variables have their FINAL SHAPE and no data in them, and
return the group so the caller can write into the arrays as it goes.

`geosave` cannot do this. It creates each variable's array with the right shape and
chunking, and then writes the whole of the caller's array into it -- so a store of
`nx × ny × nz × nt` values costs that much host memory at once, and that much disk,
before a single value has been produced. For a store that is filled a slice at a time
over days, that is the difference between a run that needs a few gigabytes and one that
needs a few hundred, and the difference between finishing and dying.

The layout is exactly the one `backend_write` produces, so `backend_open` reads a
template-created store the same way it reads a saved one:

  * the group carries the dimension record and the dataset's own attributes;
  * each declared dimension with coordinates has a 1-D coordinate array;
  * each variable has an array of the declared shape, chunked, with
    `_ARRAY_DIMENSIONS` naming its axes.

`variables` is a vector of `(name, shape, dim_names, chunks)` tuples. `chunks` may be
`nothing` for the default.
"""
function create_zarr_template(uri::AbstractString;
                              dims::Dict{Symbol, Dimension},
                              attrs::Dict{String, Any} = Dict{String, Any}(),
                              variables::Vector{<:Tuple{String, Tuple, Vector{String}, Any}})
    path = strip_scheme(uri)
    isdir(path) && rm(path; recursive = true)
    g = zgroup(path; attrs = merge(_dimension_record(dims), attrs))

    # Dimensions first: a coordinate array defines the axis its variables are chunked on.
    for (dim, d) in dims
        d.coords === nothing && continue
        zcreate(Float64, g, string(dim), length(d.coords);
                chunks = default_chunks((length(d.coords),)),
                attrs = Dict{String, Any}("units" => something(d.units, "")))
        g.arrays[string(dim)][:] = collect(Float64, d.coords)
    end

    for (name, shape, dim_names, chunks) in variables
        chosen = isnothing(chunks) ? default_chunks(shape) : Tuple(chunks)
        zcreate(Float32, g, name, shape...;
                chunks = chosen,
                attrs = Dict{String, Any}("_ARRAY_DIMENSIONS" => [String(n) for n in dim_names]))
    end

    return g
end

function backend_create(backend::ZarrBackend, uri::AbstractString,
                        dims::Dict{Symbol, Dimension}; kwargs...)
    path = strip_scheme(uri)
    if isdir(path)
        get(kwargs, :overwrite, false) || error(
            "Cannot create Zarr store '$(uri)': it already exists. Pass overwrite=true " *
            "to replace it.")
        rm(path, recursive = true)
    end
    return zgroup(path; attrs = get(kwargs, :attrs, Dict{String, Any}()))
end
"""
    _dimension_record(dims) -> Dict{String, Any}

The dataset's dimensions, as a record `backend_open` can read back.

A Zarr array's `.zarray` carries its shape, so the *existence* of an axis survives without
this; what does not survive is which axis is which, because the reader's only alternative is
to recognise a coordinate array by its name — and only the geographical names are
recognisable. A dataset whose axes are `unit`/`month`/`draw` would come back with those as
data variables, so the reader would have to know by hand which array is the spatial axis.
Recording them makes the store self-describing, and an older store without the record falls
back to the name heuristic.
"""
function _dimension_record(dims::Dict{Symbol, <:Dimension})::Dict{String, Any}
    isempty(dims) && return Dict{String, Any}()
    records = [Dict{String, Any}(
        "name" => string(d.name),
        "size" => d.size,
        # Written as the field itself rather than `something(field, nothing)`, which
        # throws when the field *is* nothing.
        "units" => d.units,
        "standard_name" => d.standard_name,
        "dim_type" => d.dim_type,
        "coords" => collect(Float64, d.coords),
    ) for d in values(dims) if d.coords !== nothing]
    isempty(records) && return Dict{String, Any}()
    return Dict{String, Any}("dimensions" => records)
end

function backend_write(backend::ZarrBackend, uri::AbstractString,
                       ds::GeoDataset; kwargs...)
    path = strip_scheme(uri)
    if isdir(path)
        # Zarr.jl reopens existing arrays read-only and refuses to create a new array in a
        # reopened group, so an existing store cannot be updated in place. Say so instead
        # of failing half-way through a write.
        get(kwargs, :overwrite, false) || error(
            "Zarr store '$(uri)' already exists. An existing store cannot be updated in " *
            "place by this backend: pass overwrite=true to rewrite it, or choose " *
            "another path.")
        rm(path; recursive = true)
    end
    g = backend_create(backend, path, ds.dims;
                       attrs = merge(_dimension_record(ds.dims), ds.attrs), overwrite = true)

    # Dimensions first: coordinate arrays define the axes the variables are chunked on. A
    # dimension with no coordinate array is not an error: a Zarr array's `.zarray` carries its
    # own shape, so the axis survives a round trip without values, and the read path already
    # reconstructs such an axis (`_zarr_dims_to_tuple` names it and leaves values unknown).
    # NetCDF's CF bounds convention creates exactly this - `climatology_bounds` over
    # `nbounds` - and refusing the whole save over it left WOA23 writing a store that held
    # nothing but `lat`.
    for (dim, d) in ds.dims
        d.coords === nothing && continue
        zcreate(Float64, g, string(dim), length(d.coords);
                chunks = default_chunks((length(d.coords),)),
                attrs = Dict{String, Any}("units" => something(d.units, "")))
        g.arrays[string(dim)][:] = collect(Float64, d.coords)
    end

    for (name, ga) in ds.variables
        sz = size(ga.data)
        names = Symbol[]
        for i in 1:length(sz)
            nm = dimension_name(ga, i)
            nm === nothing && error("Axis $i of variable $name carries no declared " *
                                     "dimension name. Declare every dimension a variable " *
                                     "spans before saving.")
            push!(names, nm)
        end
        check_declared_dims(sz, names, ds.dims)
        var_attrs = merge(Dict{String, Any}("_ARRAY_DIMENSIONS" => [string(nm) for nm in names]),
                          ga.attrs)
        var_chunks = get(ga.attrs, "chunks", get(ga.attrs, :chunks, nothing))
        chosen_chunks = isnothing(var_chunks) ? default_chunks(sz) : Tuple(var_chunks)
        zcreate(eltype(ga.data), g, name, sz...;
                chunks = chosen_chunks, attrs = var_attrs)
        g.arrays[name][:] = ga.data
    end

    return ds
end

"""
    backend_capabilities(backend::ZarrBackend)

Zarr stores are chunked and compressible, and `backend_open` hands out the chunked
arrays themselves, so reads stay lazy and slicing a loaded dataset returns views.
"""
function backend_capabilities(backend::ZarrBackend)
    BackendCapabilities(read = true, write = true, lazy = true, chunked = true)
end

backend_close(backend::ZarrBackend, ds::GeoDataset; kwargs...) = nothing
