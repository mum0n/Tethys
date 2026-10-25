"""
    geoparquet_backend.jl

GeoParquet storage backend for vector/tabular datasets.

A table is one dimension, `:points`, of length `nrow(df)`. When the geometry column
holds point-like geometries, `:lon` and `:lat` become coordinate arrays over `:points`,
so a point cloud can be sliced by coordinate like a gridded dataset. For line and
polygon geometries there are no coordinate values, so only `:points` is defined:
slicing those by coordinate is not meaningful and is never faked.

The geometry column is preserved as a variable, so a write-read round trip returns the
file it was given.
"""

using GeoParquet
using DataFrames
using GeoInterface

struct GeoParquetBackend <: GeoBackend end

function _parquet_geometry_column(df::DataFrame)
    cols = try
        collect(GeoInterface.geometrycolumns(df))
    catch
        Symbol[]
    end
    isempty(cols) || return cols[1]
    for name in (:geometry, :geom, :wkb_geometry)
        name in names(df) && return name
    end
    return nothing
end

"""
    _point_coords(g) -> (lon, lat) or nothing

Coordinates of a point-like geometry, or `nothing` when the geometry is not a point.
GeoParquet hands back well-known binary, and GeoInterface's traits work on it directly,
so no format-specific conversion is needed.
"""
function _point_coords(g)
    t = GeoInterface.trait(g)
    if t isa GeoInterface.PointTrait
        return (Float64(GeoInterface.x(g)), Float64(GeoInterface.y(g)))
    elseif t isa GeoInterface.MultiPointTrait
        parts = GeoInterface.geometries(g)
        isempty(parts) && return nothing
        return (Float64(GeoInterface.x(first(parts))), Float64(GeoInterface.y(first(parts))))
    end
    return nothing
end

"""
    _extract_points(geoms) -> (lons, lats) or nothing

Coordinates for point-like geometries, or `nothing` if any geometry is not a point.
"""
function _extract_points(geoms)
    lons = Float64[]
    lats = Float64[]
    for g in geoms
        pt = _point_coords(g)
        pt === nothing && return nothing
        push!(lons, pt[1])
        push!(lats, pt[2])
    end
    return (lons, lats)
end

function backend_open(backend::GeoParquetBackend, uri::AbstractString; kwargs...)
    path = strip_scheme(uri)
    isfile(path) || error("Cannot open GeoParquet file '$(uri)': file does not exist.")
    df = GeoParquet.read(path)
    n_points = nrow(df)
    n_points > 0 || error("Cannot open GeoParquet file '$(uri)': the table is empty.")

    crs = CoordinateSystem(crs = "EPSG:4326")
    points_dim = Dimension(name = :points, size = n_points, coords = nothing)
    dims = Dict{Symbol, Dimension}(:points => points_dim)
    coords = Dict{Symbol, GeoArray}()

    geom_col = _parquet_geometry_column(df)
    if geom_col !== nothing
        pts = _extract_points(df[!, geom_col])
        if pts !== nothing
            for (name, vals, units) in ((:lon, pts[1], "degrees_east"),
                                        (:lat, pts[2], "degrees_north"))
                d = Dimension(name = name, size = n_points, coords = vals, units = units,
                              dim_type = DIM_SPATIAL)
                dims[name] = d
                coords[name] = GeoArray(vals, (points_dim,), crs, Dict{String, Any}())
            end
        end
    end

    vars = Dict{String, GeoArray}()
    for col in names(df)
        vars[col] = GeoArray(collect(df[!, col]), (points_dim,), crs, Dict{String, Any}())
    end

    return GeoDataset(vars, coords, dims, crs, Dict{String, Any}(), backend, path)
end

"""
    backend_create(backend::GeoParquetBackend, uri, dims; kwargs...)

GeoParquet has no container to create: a table's dimensions come from its variables, and
`backend_write` creates the file. Returns the backend so callers may treat it uniformly.
"""
backend_create(backend::GeoParquetBackend, uri::AbstractString,
               dims::Dict{Symbol, Dimension}; kwargs...) = backend

function backend_write(backend::GeoParquetBackend, uri::AbstractString,
                       ds::GeoDataset; kwargs...)
    path = strip_scheme(uri)
    if isfile(path) && !get(kwargs, :overwrite, false)
        error("GeoParquet file '$(uri)' exists. Pass overwrite=true to replace it.")
    end

    columns = Dict{String, Vector}()
    geom_name = nothing
    for (name, ga) in ds.variables
        if ndims(ga) == 0
            data = [ga.data[]]
        elseif length(ga.dims) == 1 && ga.dims[1].name == :points
            data = collect(vec(ga.data))
        else
            error("Variable '$name' spans dimensions " *
                  "$([d.name for d in ga.dims]); a GeoParquet table can only hold " *
                  "1-D columns over the :points dimension.")
        end
        if name == "geometry"
            geom_name = name
        end
        columns[name] = data
    end
    isempty(columns) && error("Cannot write a GeoParquet file from a dataset with no variables.")

    names_ = collect(keys(columns))
    lengths = unique(length.(values(columns)))
    length(lengths) == 1 || error("Cannot write a GeoParquet file: columns have " *
                                  "different lengths $(lengths).")

    df = DataFrame()
    for n in names_
        df[!, n] = columns[n]
    end

    mkpath(dirname(path))
    if geom_name === nothing
        GeoParquet.write(path, df, ())
    else
        GeoParquet.write(path, df, (Symbol(geom_name),))
    end
    return ds
end

function backend_capabilities(backend::GeoParquetBackend)
    BackendCapabilities(read = true, write = true, lazy = false, chunked = false)
end

backend_close(backend::GeoParquetBackend, ds::GeoDataset; kwargs...) = nothing
