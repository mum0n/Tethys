"""
    types.jl

Core dataset, array, dimension, and coordinate reference system definitions.
"""
module Types

using Base: @kwdef

export DimensionType, DIM_SPATIAL, DIM_TEMPORAL, DIM_PARAMETRIC, DIM_GENERIC,
       Dimension, CoordinateSystem, coordinate_system_attributes, GeoArray, GeoDataset,
       GeoBackend, BackendCapabilities, is_cartesian

"""
    DimensionType

Semantic type of a dimension for generic handling.
"""
@enum DimensionType begin
    DIM_SPATIAL     # x, y, z (cartesian or geographic coordinates)
    DIM_TEMPORAL    # time coordinate
    DIM_PARAMETRIC  # parameter or ensemble index (e.g. MCMC chain, draw)
    DIM_GENERIC     # any other dimension
end

"""
    Dimension

Describes a single dimension of a dataset.

# Fields
- `name::Symbol`: Dimension identifier (e.g. `:x`, `:y`, `:lon`, `:lat`, `:depth`, `:time`)
- `size::Union{Int, Nothing}`: Dimension length
- `coords::Union{AbstractVector, Nothing}`: 1D coordinate values
- `units::Union{String, Nothing}`: Dimension units
- `standard_name::Union{String, Nothing}`: CF-compliant standard name
- `dim_type::DimensionType`: Semantic dimension category
- `calendar::Union{String, Nothing}`: Calendar for temporal dimensions
- `is_unlimited::Bool`: Whether the dimension is record/unlimited
"""
@kwdef struct Dimension
    name::Symbol
    size::Union{Int, Nothing} = nothing
    coords::Union{AbstractVector, Nothing} = nothing
    units::Union{String, Nothing} = nothing
    standard_name::Union{String, Nothing} = nothing
    dim_type::DimensionType = DIM_GENERIC
    calendar::Union{String, Nothing} = nothing
    is_unlimited::Bool = false
end

Dimension(name::Symbol, size::Int; kwargs...) =
    Dimension(; name = name, size = size, kwargs...)

function Dimension(d::Dimension; kwargs...)
    base = NamedTuple(f => getfield(d, f) for f in fieldnames(Dimension))
    return Dimension(; base..., kwargs...)
end

"""
    CoordinateSystem

Coordinate Reference System description.
"""
@kwdef struct CoordinateSystem
    crs::String = ""
    proj4::Union{String, Nothing} = nothing
    wkt::Union{String, Nothing} = nothing
end

function CoordinateSystem(attrs::AbstractDict)
    getstr(keys, default = nothing) = begin
        value = nothing
        for key in keys
            if haskey(attrs, key)
                value = attrs[key]
                break
            elseif haskey(attrs, Symbol(key))
                value = attrs[Symbol(key)]
                break
            end
        end
        isnothing(value) || isempty(strip(string(value))) ? default : string(value)
    end
    crs_value = getstr(("crs", "epsg_code"), "")
    wkt_value = getstr(("crs_wkt", "spatial_ref", "wkt"))
    if isempty(crs_value) == false && occursin(r"^(GEOGCS|PROJCS|GEODCRS|GEOGCRS|PROJCRS|COMPOUNDCRS)\s*\["i, crs_value)
        isnothing(wkt_value) || wkt_value == crs_value || throw(ArgumentError(
            "Coordinate metadata contains conflicting WKT definitions."))
        wkt_value = crs_value
        crs_value = ""
    end
    return CoordinateSystem(crs = crs_value,
                            proj4 = getstr(("proj4", "proj4text")),
                            wkt = wkt_value)
end

"""Serialize only explicitly declared CRS values into backend metadata."""
function coordinate_system_attributes(crs::CoordinateSystem)
    attrs = Dict{String, Any}()
    !isempty(crs.crs) && (attrs["crs"] = crs.crs)
    !isnothing(crs.wkt) && (attrs["crs_wkt"] = crs.wkt)
    !isnothing(crs.proj4) && (attrs["proj4"] = crs.proj4)
    return attrs
end

is_cartesian(crs::CoordinateSystem) = isempty(crs.crs) && isnothing(crs.wkt) && isnothing(crs.proj4)

"""
    GeoArray{T, N, A <: AbstractArray{T, N}}

An n-dimensional array wrapper equipped with dimension and CRS metadata.
"""
struct GeoArray{T, N, A <: AbstractArray{T, N}}
    data::A
    dims::NTuple{N, Dimension}
    crs::CoordinateSystem
    attrs::Dict{String, Any}

    function GeoArray{T, N, A}(
        data::A,
        dims::NTuple{N, Dimension},
        crs::CoordinateSystem,
        attrs::Dict{String, Any}
    ) where {T, N, A}
        length(dims) == N || error("Dimension count ($(length(dims))) must match array ndims ($N)")
        new{T, N, A}(data, dims, crs, attrs)
    end
end

function _string_attrs(attrs)
    attrs isa Dict{String, Any} && return attrs
    return Dict{String, Any}(string(k) => v for (k, v) in pairs(attrs))
end

function GeoArray(
    data::A;
    dims::Vector{Dimension} = Dimension[],
    crs::CoordinateSystem = CoordinateSystem(),
    attrs = Dict{String, Any}()
) where {T, N, A <: AbstractArray{T, N}}
    if isempty(dims)
        dims = [Dimension(name = Symbol("dim$i"), size = size(data, i)) for i in 1:N]
    end
    return GeoArray{T, N, A}(data, Tuple(dims), crs, _string_attrs(attrs))
end

function GeoArray(
    data::AbstractArray,
    dims::NTuple{N, Dimension},
    crs::CoordinateSystem,
    attrs = Dict{String, Any}()
) where {N}
    T = eltype(data)
    return GeoArray{T, N, typeof(data)}(data, dims, crs, _string_attrs(attrs))
end

Base.size(A::GeoArray) = size(A.data)
Base.size(A::GeoArray, i::Int) = size(A.data, i)
Base.ndims(A::GeoArray) = ndims(A.data)
Base.eltype(A::GeoArray{T}) where {T} = T
Base.length(A::GeoArray) = length(A.data)
Base.getindex(A::GeoArray, inds...) = A.data[inds...]
Base.setindex!(A::GeoArray, v, inds...) = (A.data[inds...] = v)
Base.IndexStyle(::Type{<:GeoArray{T, N, A_}}) where {T, N, A_} = IndexStyle(A_)
Base.similar(A::GeoArray, ::Type{T}, dims::Dims) where {T} = similar(A.data, T, dims)

"""
    GeoBackend

Abstract base type for storage backends (Zarr, NetCDF, GeoParquet).
"""
abstract type GeoBackend end

"""
    BackendCapabilities

Feature capability matrix of a storage backend.
"""
@kwdef struct BackendCapabilities
    read::Bool = true
    write::Bool = true
    lazy::Bool = false
    chunked::Bool = false
    compression::Bool = false
    parallel::Bool = false
    remote::Bool = false
    append::Bool = false
    transactions::Bool = false
end

"""
    GeoDataset

Unified collection of named `GeoArray` variables and shared coordinate dimensions.
"""
struct GeoDataset
    variables::Dict{String, GeoArray}
    coords::Dict{Symbol, GeoArray}
    dims::Dict{Symbol, Dimension}
    crs::CoordinateSystem
    attrs::Dict{String, Any}
    backend::Any
    source::String

    function GeoDataset(
        variables::Dict{String, GeoArray},
        coords::Dict{Symbol, GeoArray},
        dims::Dict{Symbol, Dimension},
        crs::CoordinateSystem,
        attrs::Dict{String, Any},
        backend::Any,
        source::String
    )
        new(variables, coords, dims, crs, attrs, backend, source)
    end
end

function GeoDataset(;
    variables::Dict = Dict{String, GeoArray}(),
    coords::Dict = Dict{Symbol, GeoArray}(),
    dims::Dict{Symbol, Dimension} = Dict{Symbol, Dimension}(),
    crs::CoordinateSystem = CoordinateSystem(),
    attrs = Dict{String, Any}(),
    backend::Any = nothing,
    source::AbstractString = ""
)
    vars = Dict{String, GeoArray}(string(k) => v for (k, v) in variables)
    coord_map = Dict{Symbol, GeoArray}(Symbol(k) => v for (k, v) in coords)
    dim_map = Dict{Symbol, Dimension}(Symbol(k) => v for (k, v) in dims)
    attr_map = _string_attrs(attrs)
    return GeoDataset(vars, coord_map, dim_map, crs, attr_map, backend, String(source))
end

function GeoDataset(
    variables::Dict,
    coords::Dict,
    dims::Dict,
    crs::CoordinateSystem,
    attrs,
    backend::Any,
    source::AbstractString
)
    vars = Dict{String, GeoArray}(string(k) => v for (k, v) in variables)
    coord_map = Dict{Symbol, GeoArray}(Symbol(k) => v for (k, v) in coords)
    dim_map = Dict{Symbol, Dimension}(Symbol(k) => v for (k, v) in dims)
    attr_map = _string_attrs(attrs)
    return GeoDataset(vars, coord_map, dim_map, crs, attr_map, backend, String(source))
end

function GeoDataset(
    variables::Dict,
    coords::Dict,
    dims::Dict;
    crs::CoordinateSystem = CoordinateSystem(),
    attrs = Dict{String, Any}(),
    backend::Any = nothing,
    source::AbstractString = ""
)
    return GeoDataset(variables, coords, dims, crs, attrs, backend, source)
end

Base.getproperty(ds::GeoDataset, name::Symbol) = begin
    if hasfield(GeoDataset, name)
        getfield(ds, name)
    elseif haskey(ds.variables, string(name))
        ds.variables[string(name)]
    elseif haskey(ds.coords, name)
        ds.coords[name]
    else
        error("Property $(name) not found in GeoDataset")
    end
end

Base.propertynames(ds::GeoDataset, private::Bool = false) = begin
    names = fieldnames(GeoDataset)
    vars = collect(keys(ds.variables))
    coords = collect(keys(ds.coords))
    return (names..., vars..., coords...)
end

Base.hasproperty(ds::GeoDataset, name::Symbol) =
    hasfield(GeoDataset, name) || haskey(ds.variables, string(name)) || haskey(ds.coords, name)

Base.getindex(ds::GeoDataset, var_name::String) = ds.variables[var_name]
Base.haskey(ds::GeoDataset, var_name::String) = haskey(ds.variables, var_name)
Base.keys(ds::GeoDataset) = (keys(ds.variables)..., keys(ds.coords)...)
Base.values(ds::GeoDataset) = (values(ds.variables)..., values(ds.coords)...)
Base.iterate(ds::GeoDataset, state = nothing) =
    iterate(pairs(merge(ds.variables, ds.coords)), state)

end # module Types
