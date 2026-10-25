"""
    backends/common.jl

Helpers shared by every backend. These exist because the same two concerns - naming a
variable's axes from its shape, and choosing default chunk sizes - were independently
reimplemented in four backend files and had drifted apart.
"""

# Canonical axis order tried first when a variable's shape must be matched against
# declared dimensions. `x, y, z` after `lon, lat, depth` so Cartesian datasets still
# resolve, and `time` is tried before the Cartesian trio because it is the axis most
# often confused with depth by size.
const STANDARD_AXIS_ORDER = (:lon, :lat, :depth, :time, :x, :y, :z)

"""
    infer_dim_names_from_size(shape, dims::Dict{Symbol, Dimension}) -> Vector{Symbol}

Map each entry of `shape` to a declared dimension name.

Dimensions are matched preferentially in `STANDARD_AXIS_ORDER`, then by any unused
dimension of the same size. A size that matches no declared dimension becomes
`dim1`, `dim2`, ... in position order. No dimension name is used twice.

Callers that write files must check the result against `dims` themselves - an inferred
name is not a declaration.
"""
function infer_dim_names_from_size(shape, dims::Dict{Symbol, <:Dimension})
    names = Symbol[]
    used = Set{Symbol}()
    for s in shape
        found = nothing
        for name in STANDARD_AXIS_ORDER
            d = get(dims, name, nothing)
            if d !== nothing && d.size == s && !(name in used)
                found = name
                break
            end
        end
        if isnothing(found)
            for (name, d) in dims
                if d.size == s && !(name in used)
                    found = name
                    break
                end
            end
        end
        if isnothing(found)
            push!(names, Symbol("dim$(length(names) + 1)"))
        else
            push!(names, found)
            push!(used, found)
        end
    end
    return names
end

"""
    default_chunks(shape) -> Tuple{Vararg{Int}}

Default chunk sizes for a new array of `shape`: capped so a chunk stays in the 100 kB -
a few MB range for float data, and every axis keeps at least one chunk.

    julia> default_chunks((10000, 100, 40))
    (1000, 100, 40)
"""
function default_chunks(shape)
    n = length(shape)
    n == 0 && return ()
    n == 1 && return (min(1000, shape[1]),)
    caps = ntuple(i -> i <= 2 ? 100 : (i == 3 ? 10 : 1), n)
    return ntuple(i -> max(1, min(caps[i], shape[i])), n)
end

"""
    check_declared_dims(shape, dim_names, dims) -> nothing

Throw unless every axis of a variable being written is covered by a declared dimension.
"""
function check_declared_dims(shape, dim_names::Vector{Symbol}, dims::Dict{Symbol, <:Dimension})
    length(dim_names) == length(shape) || error(
        "internal error: $(length(shape)) axes but $(length(dim_names)) dimension names")
    for (i, name) in enumerate(dim_names)
        d = get(dims, name, nothing)
        if d === nothing
            error("Axis $i of size $(shape[i]) was not matched by any declared dimension. " *
                  "Declared: $(sort(collect(keys(dims)))). Declare every dimension a " *
                  "variable spans before saving.")
        end
        d.size == shape[i] || error(
            "Dimension :$name is declared with size $(d.size) but the variable axis has " *
            "size $(shape[i]).")
    end
    return nothing
end
