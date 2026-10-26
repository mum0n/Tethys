"""
Generic backend-agnostic operations on GeoDataset.

These functions work on the abstract GeoDataset container regardless of
which backend produced it. Backends may override individual operations
(`backend_slice`, `backend_select`, ...) for lazy/push-down evaluation;
the default implementation lives here.
"""
module Operations

using ..Types
using ..Coordinates
using ..Backends
using ..CRS: AbstractProjection, GeographicProjection, projection_from_crs,
            transform_coordinates, projection_signature
using LinearAlgebra
using Statistics
using SparseArrays
using Interpolations

export
    slice,
    select_values,
    values_at,
    build_index,
    join_datasets,
    regrid,
    aggregate,
    dataset_summary,
    dataset_stats,
    variable_stats,
    _apply_slice,
    _gather_dim,
    _intersect_sorted,
    _union_sorted,
    _infer_dim_names,
    dimension_name

# ============================================================
# Dimension selection contract
# ============================================================

"""
    DimSelection

How a keyword argument to `slice`/`select_values` was interpreted.

- `Bounded{UnitRange{Int}}` - `dim = (lo, hi)`: the index range between the two
  coordinates, both bounds snapped to the nearest contained coordinate.
- `Bounded{Vector{Int}}` - `dim = [v1, v2, ...]`: those indices, in that order.
- `Dropped{Any}` - `dim = v`: a single coordinate value. The dimension is removed from
  the returned dataset's variables, `dims`, and `coords`; the value is recorded in
  `attrs["selections"]` so nothing is silently lost.

Every selection is resolved to indices against the dataset's coordinates at parse time,
so a dimension whose values are unsorted or duplicated fails here rather than producing
a nonsense slice.
"""
abstract type DimSelection end
struct Bounded{I} <: DimSelection
    indices::I
end
struct Dropped <: DimSelection
    index::Int
    value::Any
end

"""
    _parse_dim_specs(ds, kwargs) -> (bounded, dropped)

Resolve every keyword selection to indices against the dataset's own coordinates.

Raises for a dimension with no coordinate, a malformed range, or a value type that is
neither a range, a vector, nor a number - a typo in a keyword is never silently ignored.
"""
function _parse_dim_specs(ds::GeoDataset, kwargs)
    bounded = Dict{Symbol, DimSelection}()
    dropped = Dict{Symbol, Dropped}()
    for (k, v) in kwargs
        name = standardize_dimension_name(Symbol(k))
        if !haskey(ds.coords, name)
            error("Dataset has no coordinate for dimension :$name. Available dimensions: " *
                  "$(sort(collect(keys(ds.dims)))).")
        end
        c = vec(Float64.(ds.coords[name].data))
        isempty(c) && error("Dimension :$name has no coordinate values to select from.")
        if v isa Tuple
            length(v) == 2 || error("A range for :$name must be a (min, max) pair, got " *
                                    "$(length(v)) values.")
            bounded[name] = Bounded(slice_indices(c, Float64.(v)))
        elseif v isa AbstractVector
            isempty(v) && error("The index vector for :$name is empty.")
            bounded[name] = Bounded([find_coord_indices(c, Float64(x)) for x in v])
        elseif v isa Real
            i = find_coord_indices(c, Float64(v))
            dropped[name] = Dropped(i, c[i])
        else
            error("Invalid selection for :$name: expected (min, max), a vector of values, " *
                  "or a single number, got $(typeof(v)).")
        end
    end
    return bounded, dropped
end

# ============================================================
# Slicing
# ============================================================

"""
    slice(ds::GeoDataset; kwargs...) -> GeoDataset

Return a view of `ds` selected by coordinate range, coordinate list, or single value.

# Keywords

Per dimension:

- `dim = (lo, hi)` - the index range spanning the two coordinates. Bounds outside the
  coordinate range snap inward; an empty range is an error.
- `dim = [v1, v2, ...]` - the nearest index to each value, in that order (fancy select;
  duplicates are kept).
- `dim = v` - the nearest index, and the dimension is **dropped** from the returned
  dataset's variables, `dims`, and `coords`. The selected value is recorded in
  `attrs["selections"]`.

Dimensions not named are returned whole. The result satisfies
`assert_dataset_invariants`, so a scalar selection reduces the rank everywhere,
consistently.
"""
function slice(ds::GeoDataset; kwargs...)
    bounded, dropped = _parse_dim_specs(ds, kwargs)

    new_vars = Dict{String, GeoArray}()
    for (name, ga) in ds.variables
        data, dims = _apply_selection(ga, bounded, dropped)
        new_vars[name] = GeoArray(data, dims, ga.crs, ga.attrs)
    end

    new_coords = Dict{Symbol, GeoArray}()
    new_dims = Dict{Symbol, Dimension}()
    for (dim, ga) in ds.coords
        if haskey(bounded, dim)
            nc = vec(ga.data)[bounded[dim].indices]
            d = _with_new_coords(ga.dims[1], nc)
            new_coords[dim] = GeoArray(nc, (d,), ga.crs, ga.attrs)
            new_dims[dim] = d
        elseif haskey(dropped, dim)
            # Rank-dropping selection: the dimension leaves variables, dims and coords.
            continue
        else
            new_coords[dim] = ga
            new_dims[dim] = haskey(ds.dims, dim) ? ds.dims[dim] : ga.dims[1]
        end
    end

    attrs = copy(ds.attrs)
    if !isempty(dropped)
        selections = copy(get(ds.attrs, "selections", Dict{Symbol, Any}()))
        for (dim, sel) in dropped
            selections[dim] = sel.value
        end
        attrs["selections"] = selections
    end

    return GeoDataset(new_vars, new_coords, new_dims, ds.crs, attrs, ds.backend, ds.source)
end

function _with_new_coords(dim::Dimension, coords::AbstractVector)::Dimension
    Dimension(dim; size = length(coords), coords = collect(Float64, coords))
end

"""
    _apply_selection(ga, bounded, dropped) -> (data, dims)

Index one variable by the parsed selections, dropping selected dimensions.
"""
function _apply_selection(ga::GeoArray, bounded::Dict{Symbol, <:DimSelection},
                          dropped::Dict{Symbol, <:Dropped})
    inds = Any[]
    dims = Dimension[]
    for d in ga.dims
        sel = get(bounded, d.name, nothing)
        if sel !== nothing
            push!(inds, sel.indices)
            push!(dims, _with_new_coords(d, d.coords === nothing ? Float64[] :
                                          d.coords[sel.indices]))
        elseif haskey(dropped, d.name)
            push!(inds, dropped[d.name].index)   # scalar index: the axis is consumed
        else
            push!(inds, :)
            push!(dims, d)
        end
    end
    data = ga.data[inds...]
    if isempty(dims) && !(data isa AbstractArray)
        # Every axis collapsed: keep a 0-d array so the GeoArray stays well-formed.
        data = fill(data)
    end
    return data, tuple(dims...)
end

# ============================================================
# Selection (exact/nearest values)
# ============================================================

"""
    select_values(ds::GeoDataset; kwargs...) -> GeoDataset

Select specific coordinate values (nearest or exact match).
Like `slice` but intended for discrete value selection.
"""
function select_values(ds::GeoDataset; kwargs...)
    slice(ds; kwargs...)
end

# ============================================================
# Point value queries
# ============================================================

"""
    values_at(ds::GeoDataset, varnames::Vector{String}; kwargs...) -> Dict

Query values at specific coordinate points.

# Arguments
- `ds`: Source dataset
- `varnames`: Variables to query
- `kwargs`: Coordinate points as `lon = -65.0, lat = 45.0, depth = 50.0, time = ...`,
  each a single number, a vector of numbers, or a `(lo, hi)` pair

# Returns
Dict mapping variable name to the queried value: a scalar when every selected
dimension is dropped from that variable, an array otherwise.
"""
function values_at(ds::GeoDataset, varnames::Vector{String}; kwargs...)
    isempty(kwargs) && error("values_at requires at least one coordinate keyword, " *
                             "for example lon = -64.0, lat = 44.0.")

    # Several vectors of different lengths would silently form an outer product.
    lens = Int[]
    for (_, v) in kwargs
        v isa AbstractVector && push!(lens, length(v))
        v isa Tuple && push!(lens, 1)
    end
    if length(unique(lens)) > 1
        error("Point selections must have matching lengths, got $(lens). Query one " *
              "dimension of values at a time, or call slice for a box.")
    end

    ds_sel = slice(ds; kwargs...)
    result = Dict{String, Any}()
    for name in varnames
        haskey(ds_sel.variables, name) || error("undefined variable '$name'")
        ga = ds_sel.variables[name]
        result[name] = ndims(ga) == 0 ? ga.data[] : ga.data
    end
    return result
end

# ============================================================
# Indexing
# ============================================================

"""
    GeoIndex

Lightweight spatial/temporal index for fast subsetting.
"""
struct GeoIndex
    coords::Dict{Symbol, Vector{Float64}}
    bbox::NamedTuple
end

"""
    build_index(ds::GeoDataset; spatial::Bool=true, temporal::Bool=true) -> GeoIndex

Build a coordinate index for fast spatial/temporal queries.
"""
function build_index(ds::GeoDataset; spatial::Bool=true, temporal::Bool=true)
    coords = Dict{Symbol, Vector{Float64}}()
    for dim in (:lon, :lat, :depth, :time)
        if haskey(ds.coords, dim)
            coords[dim] = vec(Float64.(ds.coords[dim].data))
        end
    end
    bbox = bounding_box(ds)
    return GeoIndex(coords, bbox)
end

# ============================================================
# Joining
# ============================================================

"""
    dimension_name(ga::GeoArray, axis::Int) -> Symbol

Name of the dimension on `axis`, or `dim{axis}` when the array carries no dimension
metadata. Never raises: a nameless axis still needs to name itself in error messages.
"""
function dimension_name(ga::GeoArray, axis::Int)
    ga.dims === nothing && return Symbol("dim$axis")
    return ga.dims[axis].name
end

"""
    _shared_dims(dss) -> Vector{Symbol}

Dimensions declared by every dataset. Used to suggest a `join` argument.
"""
function _shared_dims(dss)
    isempty(dss) && return Symbol[]
    shared = Set{Symbol}(keys(dss[1].dims))
    for ds in dss[2:end]
        intersect!(shared, Set{Symbol}(keys(ds.dims)))
    end
    return sort!(collect(shared))
end

function _intersect_sorted(vs::Vector{Float64}...)
    isempty(vs) && return Float64[]
    result = vs[1]
    for v in vs[2:end]
        result = intersect(result, v)
        isempty(result) && break
    end
    return sort(result)
end

function _union_sorted(vs::Vector{Float64}...)
    isempty(vs) && return Float64[]
    result = vcat(vs...)
    return sort(unique(result))
end

"""
    join_datasets(dss::Vector{GeoDataset}; on::Vector{Symbol}, how::Symbol=:inner) -> GeoDataset

Join multiple datasets along shared coordinate dimensions.

Coordinate values are matched **exactly**. A coordinate value requested by `how` that a
dataset does not carry is filled with `NaN` for that dataset's variables (`:outer`,
`:left`, `:right`). Reproject or resample before joining if you want interpolation
instead of exact matching.

Variable names must be unique across the joined datasets: a collision raises instead of
being silently renamed with a suffix.

# Arguments
- `dss`: Datasets to join
- `on`: Dimension names to join on (e.g., `[:time]`, `[:lon, :lat]`)
- `how`: `:inner` (common coordinates only), `:outer` (all), `:left` (first's), `:right` (last's)

# Returns
Joined `GeoDataset`.
"""
function join_datasets(dss::Vector{GeoDataset}; on::Vector{Symbol}, how::Symbol=:inner)
    length(dss) >= 2 || error("join_datasets requires at least 2 datasets")
    how in (:inner, :outer, :left, :right) || error(
        "Unknown join type: $how (expected :inner, :outer, :left, or :right).")
    isempty(on) && error("Join on no dimension: name the dimensions explicitly, for " *
                         "example on = [:time]. Dimensions shared by all datasets: " *
                         "$(_shared_dims(dss)).")

    # A suffixed name would make downstream lookup ambiguous, so refuse the collision.
    origin = Dict{String, Int}()
    for (i, ds) in enumerate(dss)
        for name in keys(ds.variables)
            if haskey(origin, name)
                error("Variable '$name' occurs in datasets $(origin[name]) and $i; " *
                      "rename the variables before joining.")
            end
            origin[name] = i
        end
    end

    # Target coordinates for the join dimensions.
    target = Dict{Symbol, Vector{Float64}}()
    for dim in on
        for (i, ds) in enumerate(dss)
            haskey(ds.coords, dim) || error(
                "Dataset $i has no coordinate for :$dim, so the datasets cannot be " *
                "joined on it. Joining on requires every dataset to declare that " *
                "coordinate; use geojoin with a different `on` for a schema-level union.")
        end
        allvecs = [vec(Float64.(ds.coords[dim].data)) for ds in dss]
        if how === :inner
            target[dim] = _intersect_sorted(allvecs...)
        elseif how === :outer
            target[dim] = _union_sorted(allvecs...)
        elseif how === :left
            target[dim] = sort(allvecs[1])
        else  # :right
            target[dim] = sort(allvecs[end])
        end
        isempty(target[dim]) && error(
            "Joining on :$dim with how = $how leaves no coordinate values. For an " *
            "inner join the datasets share no :$dim value.")
    end

    all_vars = Dict{String, GeoArray}()
    all_coords = Dict{Symbol, GeoArray}()
    all_dims = Dict{Symbol, Dimension}()

    # Reference metadata for the joined dimensions comes from the first dataset that
    # declares them, so units and names survive the join.
    for dim in on
        ref_ds = dss[findfirst(ds -> haskey(ds.coords, dim), dss)]
        ref = ref_ds.coords[dim]
        d = _with_new_coords(ref.dims[1], target[dim])
        all_coords[dim] = GeoArray(target[dim], (d,), ref.crs, ref.attrs)
        all_dims[dim] = d
    end

    # Dimensions that are not joined are carried over from the first dataset that has
    # them; they must agree across datasets on size, or the join is ill-defined.
    for ds in dss
        for (dim, d) in ds.dims
            haskey(all_dims, dim) && continue
            all_dims[dim] = d
            if haskey(ds.coords, dim)
                all_coords[dim] = ds.coords[dim]
            elseif haskey(dss[1].coords, dim)
                all_coords[dim] = dss[1].coords[dim]
            end
        end
    end

    for (i, ds) in enumerate(dss)
        for (name, ga) in ds.variables
            data, dims = _gather_to_target(ga, target, on, ds)
            all_vars[name] = GeoArray(data, dims, ga.crs, ga.attrs)
        end
    end

    attrs = copy(dss[1].attrs)
    for ds in dss[2:end]
        merge!(attrs, ds.attrs)
    end
    attrs["joined_from"] = [ds.source for ds in dss]

    return GeoDataset(all_vars, all_coords, all_dims, dss[1].crs, attrs, dss[1].backend,
                      join([ds.source for ds in dss], "+"))
end

function _gather_to_target(ga::GeoArray, target::Dict{Symbol, Vector{Float64}},
                           on::Vector{Symbol}, source::GeoDataset)
    inds = Any[]
    dims = Dimension[]
    gaps = Pair{Int, BitVector}[]
    for d in 1:ndims(ga)
        name = dimension_name(ga, d)
        if name in on
            src_ga = get(source.coords, name, nothing)
            src_ga === nothing && error(
                "Variable spans :$name, which is a join dimension, but its dataset has " *
                "no :$name coordinate. Declare the coordinate before joining.")
            src = vec(Float64.(src_ga.data))
            lookup = Dict(src[i] => i for i in eachindex(src))
            tc = target[name]
            idx = ones(Int, length(tc))
            found = falses(length(tc))
            for (j, t) in enumerate(tc)
                i = get(lookup, t, nothing)
                if i !== nothing
                    idx[j] = i
                    found[j] = true
                end
            end
            push!(inds, idx)
            push!(gaps, d => found)
            push!(dims, _with_new_coords(ga.dims[d], tc))
        else
            push!(inds, :)
            push!(dims, ga.dims[d])
        end
    end

    data = ga.data[inds...]
    for (axis, found) in gaps
        if !all(found)
            data = _fill_missing_axis(data, axis, found)
        end
    end
    return data, tuple(dims...)
end

"""
    _fill_missing_axis(data, axis, found) -> Array

Copy `data` with `NaN` in the positions of `axis` where `found` is false. Used to fill
join gaps rather than silently duplicating the nearest neighbour.
"""
function _fill_missing_axis(data::AbstractArray, axis::Int, found::BitVector)
    out = similar(data, float(eltype(data)))
    copyto!(out, data)
    lead = ntuple(_ -> Colon(), axis - 1)
    trail = ntuple(_ -> Colon(), ndims(out) - axis)
    for j in eachindex(found)
        if !found[j]
            out[lead..., j, trail...] .= NaN
        end
    end
    return out
end

# ============================================================
# Regridding
# ============================================================

"""
    regrid(ds::GeoDataset, target::GeoDataset; method::Symbol=:bilinear, vars::Vector{String}=String[]) -> GeoDataset

Regrid dataset to a target grid.

# Arguments
- `ds`: Source dataset
- `target`: Dataset defining target coordinates
- `method`: Interpolation method (`:bilinear`, `:nearest`)
- `vars`: Variables to regrid (empty = all data variables)

# Returns
Regridded `GeoDataset`.
"""
function regrid(ds::GeoDataset, target::GeoDataset; method::Symbol=:bilinear, vars::Vector{String}=String[])
    method in (:bilinear, :nearest) || throw(ArgumentError(
        "Unsupported regrid method $(repr(method)); choose :bilinear or :nearest."))
    isempty(vars) && (vars = collect(keys(ds.variables)))

    target_coords = Dict{Symbol, Vector{Float64}}()
    for (name, coord) in target.coords
        canonical = standardize_dimension_name(name)
        canonical in (:lon, :lat, :depth, :time) || continue
        target_coords[canonical] = vec(Float64.(coord.data))
    end
    for (name, dim) in target.dims
        canonical = standardize_dimension_name(name)
        canonical in (:lon, :lat, :depth, :time) || continue
        if !haskey(target_coords, canonical) && dim.coords !== nothing
            target_coords[canonical] = vec(Float64.(dim.coords))
        end
    end

    source_projection = _dataset_projection(ds)
    target_projection = _dataset_projection(target)
    reprojecting = projection_signature(source_projection) != projection_signature(target_projection)

    new_vars = Dict{String, GeoArray}()
    for name in vars
        haskey(ds.variables, name) || throw(ArgumentError("Dataset has no variable $(repr(name))."))
        ga = ds.variables[name]
        new_data = reprojecting ?
            _regrid_array_between_crs(ga, ds, target, source_projection, target_projection,
                                      target_coords, method) :
            _regrid_array(ga, target_coords, method)
        new_dims = _build_dims_from_target(ga, target_coords, target.dims)
        new_vars[name] = GeoArray(new_data, new_dims, target.crs, ga.attrs)
    end

    return GeoDataset(new_vars, target.coords, target.dims, target.crs, ds.attrs, ds.backend, ds.source)
end

function _dataset_projection(ds::GeoDataset)
    crs = ds.crs
    if isempty(strip(crs.crs)) && (isnothing(crs.wkt) || isempty(strip(crs.wkt))) &&
       (isnothing(crs.proj4) || isempty(strip(crs.proj4)))
        _, lon_dim = axis(ds, :lon)
        _, lat_dim = axis(ds, :lat)
        lon_units = lowercase(strip(something(lon_dim.units, "")))
        lat_units = lowercase(strip(something(lat_dim.units, "")))
        lon_units in ("degrees_east", "degree_east", "degrees east") &&
        lat_units in ("degrees_north", "degree_north", "degrees north") ||
            throw(ArgumentError("Dataset $(repr(ds.source)) has no declared CRS. Provide CRS/WKT " *
                                "metadata; only explicit CF longitude/latitude degree units " *
                                "identify EPSG:4326."))
        return GeographicProjection()
    end
    return projection_from_crs(crs)
end

function _build_dims_from_target(
    ga::GeoArray,
    target_coords::Dict{Symbol, Vector{Float64}},
    target_dims::Dict{Symbol, Dimension} = Dict{Symbol, Dimension}()
)
    new_dims = Dimension[]
    for dim in ga.dims
        canonical = standardize_dimension_name(dim.name)
        target_name = findfirst(k -> standardize_dimension_name(k) == canonical,
                                collect(keys(target_dims)))
        if haskey(target_coords, canonical)
            tc = target_coords[canonical]
            target_dim = target_name === nothing ? nothing : target_dims[collect(keys(target_dims))[target_name]]
            if target_dim === nothing
                push!(new_dims, Dimension(name=dim.name, size=length(tc), coords=tc,
                                          units=dim.units, standard_name=dim.standard_name,
                                          calendar=dim.calendar, is_unlimited=dim.is_unlimited))
            else
                push!(new_dims, Dimension(name=target_dim.name, size=length(tc), coords=tc,
                                          units=target_dim.units, standard_name=target_dim.standard_name,
                                          calendar=target_dim.calendar,
                                          is_unlimited=target_dim.is_unlimited))
            end
        else
            push!(new_dims, dim)
        end
    end
    return tuple(new_dims...)
end

function _regrid_array(ga::GeoArray, target_coords::Dict{Symbol, Vector{Float64}}, method::Symbol)
    data = ga.data
    for (i, dim) in enumerate(ga.dims)
        canonical = standardize_dimension_name(dim.name)
        if haskey(target_coords, canonical) && !isnothing(dim.coords) && !isempty(dim.coords)
            src = vec(Float64.(dim.coords))
            tgt = target_coords[canonical]
            if method === :nearest
                data = _resample_nearest(data, i, src, tgt)
            else
                data = _resample_linear(data, i, src, tgt)
            end
        end
    end
    return data
end

function _regrid_array_between_crs(ga, source, target, source_projection,
                                   target_projection, target_coords, method)
    source_x, _ = axis(source, :lon)
    source_y, _ = axis(source, :lat)
    target_x = get(target_coords, :lon, nothing)
    target_y = get(target_coords, :lat, nothing)
    (target_x === nothing || target_y === nothing) && throw(ArgumentError(
        "CRS-changing regrids require both horizontal target axes (:lon/:lat or :x/:y)."))
    ix, iy = dim_permutation(ga.dims, :lon, :lat)
    permutation = vcat([ix, iy], [i for i in eachindex(ga.dims) if i != ix && i != iy])
    ordered = permutedims(Array(ga.data), permutation)
    size(ordered, 1) == length(source_x) && size(ordered, 2) == length(source_y) ||
        throw(DimensionMismatch("Variable $(repr(ga)) horizontal dimensions do not match source axes."))

    mapped_x = Matrix{Float64}(undef, length(target_x), length(target_y))
    mapped_y = similar(mapped_x)
    for j in eachindex(target_y), i in eachindex(target_x)
        mapped_x[i, j], mapped_y[i, j] = transform_coordinates(
            target_projection, source_projection, target_x[i], target_y[j])
    end

    source_3d = reshape(ordered, length(source_x), length(source_y), :)
    output_3d = Array{eltype(ordered)}(undef, length(target_x), length(target_y), size(source_3d, 3))
    for k in axes(source_3d, 3), j in eachindex(target_y), i in eachindex(target_x)
        output_3d[i, j, k] = _sample_rectilinear(
            source_3d[:, :, k], source_x, source_y, mapped_x[i, j], mapped_y[i, j], method)
    end
    ordered_result = reshape(output_3d, (length(target_x), length(target_y), size(ordered)[3:end]...))
    result = permutedims(ordered_result, invperm(permutation))

    horizontal_coords = Dict(:lon => target_x, :lat => target_y)
    result_dims = _build_dims_from_target(ga, horizontal_coords, target.dims)
    remaining_coords = Dict{Symbol, Vector{Float64}}(
        k => v for (k, v) in target_coords if k ∉ (:lon, :lat))
    isempty(remaining_coords) && return result
    return _regrid_array(GeoArray(result, result_dims, ga.crs, ga.attrs), remaining_coords, method)
end

function _sample_rectilinear(field, xs, ys, x, y, method)
    issorted(xs) && issorted(ys) || throw(ArgumentError(
        "Source CRS axes must be monotonically ascending before regridding."))
    if method === :nearest
        return field[findmin(abs.(xs .- x))[2], findmin(abs.(ys .- y))[2]]
    end
    ix = clamp(searchsortedlast(xs, x), 1, max(length(xs) - 1, 1))
    iy = clamp(searchsortedlast(ys, y), 1, max(length(ys) - 1, 1))
    length(xs) >= 2 && length(ys) >= 2 || throw(ArgumentError(
        "Bilinear CRS regridding requires at least two points on each source axis."))
    x0, x1 = xs[ix], xs[ix + 1]
    y0, y1 = ys[iy], ys[iy + 1]
    fx = clamp((x - x0) / (x1 - x0), 0.0, 1.0)
    fy = clamp((y - y0) / (y1 - y0), 0.0, 1.0)
    return (1 - fx) * (1 - fy) * field[ix, iy] +
           fx * (1 - fy) * field[ix + 1, iy] +
           (1 - fx) * fy * field[ix, iy + 1] +
           fx * fy * field[ix + 1, iy + 1]
end

function _resample_linear(data, dim::Int, src::Vector{Float64}, tgt::Vector{Float64})
    d = ndims(data)
    perm = (dim, setdiff(1:d, dim)...)
    dp = permutedims(data, perm)
    nout = length(tgt)
    tail_size = size(dp)[2:end]
    out = Array{eltype(data)}(undef, nout, tail_size...)
    for I in Iterators.product(ntuple(_ -> 1, d-1)...)
        idxs = Tuple([1] + [I...])
        line = view(dp, :, ntuple(k -> idxs[k+1], d-1)...)
        out[:, ntuple(k -> idxs[k+1], d-1)...] = _interp1d(line, src, tgt)
    end
    invperm = zeros(Int, d)
    for (i, p) in enumerate(perm)
        invperm[p] = i
    end
    return permutedims(out, invperm)
end

function _resample_nearest(data, dim::Int, src::Vector{Float64}, tgt::Vector{Float64})
    idxs = [find_coord_indices(src, t) for t in tgt]
    return _gather_dim(data, dim, idxs)
end

function _interp1d(line::AbstractVector, src::Vector{Float64}, tgt::Vector{Float64})
    n = length(src)
    n == 0 && error("Source coordinate array for interpolation cannot be empty.")
    if n == 1
        return fill(line[1], length(tgt))
    end
    if !issorted(src)
        error("Source coordinate array must be monotonically sorted for interpolation.")
    end

    y = collect(Float64, line)
    itp = extrapolate(interpolate((src,), y, Gridded(Linear())), Flat())
    return eltype(line)[itp(t) for t in tgt]
end

function _gather_dim(data, dim::Int, idxs::Vector{Int})
    d = ndims(data)
    perm = (dim, setdiff(1:d, dim)...)
    dp = permutedims(data, perm)
    sub = dp[idxs, ntuple(_ -> (:), d-1)...]
    invperm = zeros(Int, d)
    for (i, p) in enumerate(perm)
        invperm[p] = i
    end
    return permutedims(sub, invperm)
end

# ============================================================
# Aggregation
# ============================================================

"""
    aggregate(ds::GeoDataset; dim::Symbol, func::Function) -> GeoDataset

Reduce every variable along `dim` and return a dataset with that dimension removed.

`func` must accept the array-positional form `func(data; dims=i)`, so `mean`, `sum`,
`maximum`, and `extrema` all work. A variable that does not span `dim` is returned
unchanged - aggregating it is not defined, and dropping it would lose data.

The reduced dimension is removed from `variables`, `dims`, and `coords` together, and
its reduction is recorded in `attrs["reduced"]` so the output remains self-describing.
"""
function aggregate(ds::GeoDataset; dim::Symbol, func::Function)
    haskey(ds.dims, dim) || error(
        "Cannot aggregate: dataset has no dimension :$dim. Dimensions: " *
        "$(sort(collect(keys(ds.dims)))).")

    new_vars = Dict{String, GeoArray}()
    for (name, ga) in ds.variables
        dim_idx = findfirst(d -> d.name == dim, ga.dims)
        if dim_idx === nothing
            new_vars[name] = ga
            continue
        end
        new_data = dropdims(func(ga.data; dims=dim_idx); dims=dim_idx)
        new_dims = Tuple(d for (i, d) in enumerate(ga.dims) if i != dim_idx)
        new_vars[name] = GeoArray(new_data, new_dims, ga.crs, ga.attrs)
    end

    new_coords = Dict{Symbol, GeoArray}(k => v for (k, v) in ds.coords if k != dim)
    new_dims = Dict{Symbol, Dimension}(k => v for (k, v) in ds.dims if k != dim)

    attrs = copy(ds.attrs)
    reduced = copy(get(ds.attrs, "reduced", Dict{Symbol, Any}()))
    reduced[dim] = func
    attrs["reduced"] = reduced

    return GeoDataset(new_vars, new_coords, new_dims, ds.crs, attrs, ds.backend, ds.source)
end

# ============================================================
# Summary statistics
# ============================================================

"""
    dataset_stats(arr::AbstractArray) -> NamedTuple

Compute basic statistics for an array (ignoring NaN/missing).
"""
function dataset_stats(arr::AbstractArray)
    vals = collect(vec(Float64.(arr)))
    vals = vals[isfinite.(vals)]
    isempty(vals) && return (min=NaN, max=NaN, mean=NaN, std=NaN, count=0)
    return (min=minimum(vals), max=maximum(vals), mean=mean(vals), std=std(vals), count=length(vals))
end

"""
    variable_stats(ds::GeoDataset, varname::String) -> NamedTuple

Stats for a single variable.
"""
function variable_stats(ds::GeoDataset, varname::String)
    haskey(ds.variables, varname) || error("Unknown variable: $varname")
    return dataset_stats(ds.variables[varname].data)
end

"""
    dataset_summary(ds::GeoDataset) -> String

Human-readable summary of dataset.
"""
function dataset_summary(ds::GeoDataset)
    lines = String[]
    push!(lines, "GeoDataset: $(ds.source)")
    push!(lines, "Backend: $(ds.backend)")
    push!(lines, "Dimensions:")
    for (dim, d) in ds.dims
        push!(lines, "  $dim: $(d.size) points, units=$(d.units)")
    end
    push!(lines, "Variables:")
    for (name, ga) in ds.variables
        st = dataset_stats(ga.data)
        push!(lines, "  $name: $(size(ga.data)), min=$(st.min), max=$(st.max), mean=$(st.mean), count=$(st.count)")
    end
    return join(lines, "\n")
end

end # module Operations
