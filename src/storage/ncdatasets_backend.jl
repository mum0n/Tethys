"""
    ncdatasets_backend.jl

NetCDF and NetCDF-Zarr storage backend (read/modify/create).

Axes are self-describing here: the file declares each variable's dimensions, so a
1-D data variable is never mistaken for a coordinate. Time axes that carry CF
`units` are decoded to numeric values, as documented on `_read_coord`.
"""

using NCDatasets

"""
    NCDatasetsBackend(; format=nothing)

NetCDF backend. `format` selects the container NCDatasets writes (`"netcdf4"` by
default, `"nczarr"` for a Zarr-backed NetCDF store).
"""
struct NCDatasetsBackend <: GeoBackend
    format::String
    NCDatasetsBackend(; format::Union{Nothing, AbstractString} = nothing) = new(something(format, "netcdf4"))
end

function backend_open(backend::NCDatasetsBackend, uri::AbstractString;
                      format::Union{Nothing, AbstractString} = nothing, kwargs...)
    path = strip_scheme(uri)
    mode = haskey(kwargs, :mode) ? kwargs[:mode] : "r"
    fmt = something(format, _infer_format(path, backend))
    NCDataset(path, mode; format = fmt) do ds
        crs = CoordinateSystem(crs = something(get(ds.attrib, "crs", nothing), "EPSG:4326"))
        coords = Dict{Symbol, GeoArray}()
        dims = Dict{Symbol, Dimension}()
        vars = Dict{String, GeoArray}()

        # Coordinate variables: the 1-D variable whose name is its own dimension.
        for (vname, v) in ds
            if ndims(v) != 1 || NCDatasets.dimnames(v)[1] != vname
                continue
            end
            canon = standardize_dimension_name(Symbol(vname))
            val = _read_coord(v, vname)
            d = Dimension(name = canon, size = length(val), coords = val,
                          units = get(v.attrib, "units", nothing),
                          standard_name = get(v.attrib, "standard_name", nothing),
                          calendar = get(v.attrib, "calendar", nothing))
            dims[canon] = d
            coords[canon] = GeoArray(val, (d,), crs, Dict(v.attrib))
        end

        for (vname, v) in ds
            if ndims(v) == 1 && NCDatasets.dimnames(v)[1] == vname
                continue   # already read as a coordinate
            end
            vars[vname] = GeoArray(_read_data(v), _dims_for(size(v), v, dims), crs,
                                   Dict(v.attrib))
        end

        return GeoDataset(vars, coords, dims, crs, Dict(ds.attrib), backend, path)
    end
end

function _infer_format(path::AbstractString, backend::NCDatasetsBackend)
    return _default_format(backend)
end

"""
    _default_format(backend) -> String

Container format for a new file. NetCDF-4 is used because it supports groups, strings,
compression, and unlimited dimensions; the NCZarr backend-builder switches to `nczarr`.
"""
_default_format(backend::NCDatasetsBackend) = Symbol(something(backend.format, "netcdf4"))

"""
    _dims_for(shape, v, dims) -> Tuple{Vararg{Dimension}}

Dimension records for one variable of shape `shape`, taken from the file's own
`dimnames` when available and from shape matching otherwise.
"""
function _dims_for(shape::Tuple, v, dims::Dict{Symbol, Dimension})
    names = try
        map(Symbol, NCDatasets.dimnames(v))
    catch
        infer_dim_names_from_size(shape, dims)
    end
    records = Dimension[]
    for (i, name) in enumerate(names)
        canon = standardize_dimension_name(name)
        d = get(dims, canon, nothing)
        if d === nothing
            d = Dimension(name = canon, size = shape[i], coords = nothing)
            dims[canon] = d
        end
        push!(records, Dimension(d; size = shape[i]))
    end
    return tuple(records...)
end

"""
    _read_coord(v, vname) -> Vector{Float64}

Read a coordinate variable as numeric values.

Numeric axes are passed through in the file's own units - no axis convention is
imposed at load time, so what comes back is what the file says. A variable NCDatasets
decoded into dates (because its `units` say "days/hours since ...") is expressed in unix
seconds instead, since a `DateTime` has no place in a file-neutral coordinate array. A
calendar that cannot be represented as a `DateTime` raises: a silently wrong date is
worse than an error.
"""
function _read_coord(v, vname::AbstractString)
    raw = v[:]
    isempty(raw) && return Float64[]
    eltype(raw) <: Number && return collect(Float64, raw)

    # A coordinate the file decoded into dates is expressed in unix seconds, since a
    # `DateTime` has no place in a file-neutral coordinate array. Missing coordinates
    # become NaN rather than an error: WOA23's `time` axis carries a `missing` at its head
    # in the shared-risk feed, and `collect(Float64, ...)` on `Union{Missing, Date}` raised
    # before it. An axis with no valid values is a corrupt file and still errors.
    try
        vals = if eltype(raw) <: Union
            map(raw) do t
                ismissing(t) && return NaN
                t isa Dates.AbstractTime || error("not a DateTime")
                Dates.value(Dates.DateTime(t)) / 1000.0
            end
        else
            map(raw) do t
                t isa Dates.AbstractTime || error("not a DateTime")
                Dates.value(Dates.DateTime(t)) / 1000.0
            end
        end
        return Float64.(vals)
    catch
        error("Could not read coordinate '$vname' as numeric values: the file's " *
              "calendar cannot be represented as a DateTime. Read the variable with " *
              "NCDatasets directly, or resample the axis onto a numeric time base.")
    end
end

"""
    _read_data(v) -> Array{Float64}

Read a data variable into its declared shape, mapping missing values to `NaN`.

NCDatasets' `v[:]` returns a flat vector even for N-dimensional variables, so the result
is reshaped to `size(v)`; keeping the shape is what makes the dimension records line up
with the data.
"""
function _read_data(v)
    raw = collect(v[:])
    # Decided by the element type, not by `NCDatasets.ismissing`. `ismissing` asks whether
    # the variable declares `missing_value`, and WOA23's `t_an` declares `fillvalue` with an
    # empty `missing_values` attribute: NCDatasets still types the array
    # `Union{Missing, Float32}` and fills those cells with `missing`, `ismissing` says no, and
    # the conversion below then fails on the first filled cell. The element type cannot lie.
    if Missing <: eltype(raw) || eltype(raw) <: Missing
        values = map(raw) do x
            ismissing(x) ? NaN : Float64(x)
        end
    else
        values = map(raw) do x
            Float64(x)
        end
    end
    return reshape(values, size(v))
end

function backend_create(backend::NCDatasetsBackend, uri::AbstractString,
                        dims::Dict{Symbol, Dimension}; kwargs...)
    path = strip_scheme(uri)
    fmt = something(haskey(kwargs, :format) ? kwargs[:format] : nothing,
                    _default_format(backend))
    mode = haskey(kwargs, :mode) ? kwargs[:mode] : "c"
    overwrite = get(kwargs, :overwrite, false)
    if isfile(path) && overwrite
        rm(path)
    end
    return NCDataset(path, mode; format = fmt)
end

# NetCDF attributes hold numbers, strings, chars and arrays of those -- but there is no
# NetCDF `Bool`, and NCDatasets maps an attribute value's *type* to an `nc_type`, so writing
# a `Bool` (or a `Symbol`, tuple, or dict) raises `KeyError(:Bool)` and aborts the whole
# write. A producer is free to attach such metadata, so coerce it here, generically, rather
# than requiring every caller to know what NCDatasets cannot serialise.
_nc_attrib_value(v::Bool) = string(v)
_nc_attrib_value(v::Symbol) = string(v)
_nc_attrib_value(v::Union{Tuple, AbstractDict}) = string(v)
_nc_attrib_value(v::AbstractVector{Bool}) = string.(v)
_nc_attrib_value(v::AbstractVector{Symbol}) = string.(v)
_nc_attrib_value(v) = v

function backend_write(backend::NCDatasetsBackend, uri::AbstractString,
                       ds::GeoDataset; kwargs...)
    path = strip_scheme(uri)
    fmt = something(haskey(kwargs, :format) ? kwargs[:format] : nothing,
                     _default_format(backend))
    overwrite = get(kwargs, :overwrite, false)
    exists = isfile(path)
    if exists && overwrite
        rm(path)
        exists = false
    end

    ds_file = NCDataset(path, exists ? "a" : "c"; format = fmt)
    try
        for (dim, d) in ds.dims
            # `ds_file.dim` and `ds_file` are keyed by String, so the guard must pass one:
            # `haskey(ds_file.dim, dim)` with `dim` a Symbol is false for a dimension that is
            # already there, and redefining it makes netCDF-c fail with -42 ("name in use")
            # on the second `geosave` to the same path.
            if !haskey(ds_file.dim, string(dim))
                defDim(ds_file, dim, d.is_unlimited ? Inf : d.size)
            end
            if d.coords !== nothing && !haskey(ds_file, string(dim))
                defVar(ds_file, dim, Float64, (dim,),
                       attrib = Dict("units" => something(d.units, "")))
                ds_file[dim][:] = collect(Float64, d.coords)
            end
        end

        for (name, ga) in ds.variables
            sz = size(ga.data)
            axes = [dimension_name(ga, i) for i in 1:length(sz)]
            check_declared_dims(sz, axes, ds.dims)
            if !haskey(ds_file, name)
                var_attrib = Dict(k => _nc_attrib_value(v) for (k, v) in ga.attrs)
                defVar(ds_file, name, eltype(ga.data), tuple(axes...); attrib = var_attrib)
            end
            ds_file[name][ntuple(_ -> Colon(), ndims(ga))...] = ga.data
        end

        for (k, v) in ds.attrs
            ds_file.attrib[k] = _nc_attrib_value(v)
        end
    finally
        close(ds_file)
    end
    return ds
end

"""
    backend_capabilities(backend::NCDatasetsBackend)

NetCDF files are read and written through NCDatasets, which materialises variables on
`getindex`; the backend does not expose lazy views, so `lazy` is false here.
"""
function backend_capabilities(backend::NCDatasetsBackend)
    BackendCapabilities(read = true, write = true, lazy = false, chunked = true)
end
