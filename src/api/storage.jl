"""
    storage.jl

Hierarchical analytical and trajectory storage abstraction for Tethys.

Provides a unified interface (`GeoStorage`) for managing multi-group, multi-dataset
scientific archives (such as simulation runs, particle trajectories, metrics,
and demographic connectivity matrices) across both chunked tensor arrays (Zarr)
and tabular partitioned collections (GeoParquet).
"""

using ..Types
using DataFrames
using Dates
import Zarr
import GeoParquet
import TOML

export GeoStorage,
       open_geostorage,
       close_geostorage,
       create_storage_group,
       has_storage_group,
       write_storage_variable!,
       read_storage_variable,
       list_storage_variables,
       list_storage_groups,
       lakehouse_dir,
       set_lakehouse_dir!

"""
    GeoStorage

Abstract container representing a hierarchical analytical storage archive.
Supports both Zarr multi-group stores and GeoParquet directory collections.
"""
mutable struct GeoStorage
    uri::String
    backend::Symbol
    read_only::Bool
    handle::Any

    function GeoStorage(
        uri::AbstractString;
        backend::Union{Symbol, Nothing} = nothing,
        read_only::Bool = false,
        create::Bool = true
    )
        resolved_be = if !isnothing(backend)
            backend
        elseif endswith(lowercase(uri), ".parquet")
            :geoparquet
        elseif endswith(lowercase(uri), ".zarr")
            :zarr
        else
            :zarr
        end

        norm_path = if resolved_be == :zarr
            endswith(uri, ".zarr") ? String(uri) : String(uri) * ".zarr"
        elseif resolved_be == :geoparquet
            endswith(uri, ".parquet") ? String(uri) : String(uri) * ".parquet"
        else
            String(uri)
        end

        h = if resolved_be == :zarr
            _init_zarr_storage(norm_path, read_only, create)
        elseif resolved_be == :geoparquet
            _init_parquet_storage(norm_path, read_only, create)
        else
            error("Unsupported storage backend: $(resolved_be). Supported: :zarr, :geoparquet")
        end

        return new(norm_path, resolved_be, read_only, h)
    end
end

function _init_zarr_storage(path::String, read_only::Bool, create::Bool)
    if read_only
        return Zarr.zopen(path, "r")
    end
    if isdir(path) && isfile(joinpath(path, ".zgroup"))
        return Zarr.zopen(path, "w")
    else
        create && mkpath(path)
        return Zarr.zgroup(path)
    end
end

mutable struct ParquetDirectoryHandle
    path::String
    read_only::Bool
    cached_tables::Dict{String, Any}
end

function _init_parquet_storage(path::String, read_only::Bool, create::Bool)
    base_dir = if isdir(path)
        path
    elseif endswith(path, ".parquet")
        d = dirname(path)
        isempty(d) ? "." : d
    else
        path
    end
    if create && !read_only
        mkpath(base_dir)
    end
    return ParquetDirectoryHandle(path, read_only, Dict{String, Any}())
end

"""
    open_geostorage(uri::AbstractString; backend=:auto, read_only=false, create=true) -> GeoStorage

Open or initialize a hierarchical storage container at `uri`.
"""
function open_geostorage(
    uri::AbstractString;
    backend::Union{Symbol, Nothing} = nothing,
    read_only::Bool = false,
    create::Bool = true
)::GeoStorage
    return GeoStorage(uri; backend = backend, read_only = read_only, create = create)
end

"""
    close_geostorage(storage::GeoStorage) -> Nothing

Flush buffered data and release file locks.
"""
function close_geostorage(storage::GeoStorage)
    GC.gc()
    return nothing
end

"""
    create_storage_group(storage::GeoStorage, group_path::AbstractString)

Create a nested group hierarchy in the storage archive.
"""
function create_storage_group(storage::GeoStorage, group_path::AbstractString)
    if storage.backend == :zarr
        grp = storage.handle
        parts = split(replace(group_path, "\\" => "/"), "/")
        cur = grp
        for p in parts
            s = String(p)
            isempty(s) && continue
            cur = haskey(cur, s) ? cur[s] : Zarr.zgroup(cur, s)
        end
        return cur
    elseif storage.backend == :geoparquet
        target_dir = joinpath(dirname(storage.uri), group_path)
        mkpath(target_dir)
        return target_dir
    end
end

"""
    has_storage_group(storage::GeoStorage, group_path::AbstractString) -> Bool
"""
function has_storage_group(storage::GeoStorage, group_path::AbstractString)::Bool
    if storage.backend == :zarr
        parts = split(replace(group_path, "\\" => "/"), "/")
        cur = storage.handle
        for p in parts
            s = String(p)
            isempty(s) && continue
            !haskey(cur, s) && return false
            cur = cur[s]
        end
        return true
    elseif storage.backend == :geoparquet
        target_dir = joinpath(dirname(storage.uri), group_path)
        return isdir(target_dir)
    end
end

"""
    write_storage_variable!(storage::GeoStorage, var_path::AbstractString, data; kwargs...)

Write an array, vector, or table into `storage` at `var_path`.
"""
function write_storage_variable!(
    storage::GeoStorage,
    var_path::AbstractString,
    data;
    chunks = nothing,
    compressor = nothing
)
    storage.read_only && error("Cannot write to read-only GeoStorage at $(storage.uri)")

    if storage.backend == :zarr
        norm = replace(var_path, "\\" => "/")
        parts = split(norm, "/")
        var_name = String(last(parts))
        parent_group = storage.handle
        if length(parts) > 1
            for p in parts[1:end-1]
                s = String(p)
                isempty(s) && continue
                parent_group = haskey(parent_group, s) ?
                    parent_group[s] : Zarr.zgroup(parent_group, s)
            end
        end

        if haskey(parent_group, var_name)
            parent_group[var_name][:] = data
        else
            c_size = isnothing(chunks) ? size(data) : chunks
            kw = Dict{Symbol, Any}(:chunks => c_size)
            if !isnothing(compressor)
                kw[:compressor] = compressor
            end
            arr = Zarr.zcreate(
                eltype(data), parent_group, var_name, size(data)...;
                kw...
            )
            arr[:] = data
        end
        return nothing
    elseif storage.backend == :geoparquet
        target_file = if endswith(var_path, ".parquet")
            joinpath(dirname(storage.uri), var_path)
        else
            joinpath(dirname(storage.uri), var_path * ".parquet")
        end
        mkpath(dirname(target_file))
        if data isa DataFrame
            GeoParquet.write(target_file, data)
        else
            df = DataFrame(value = vec(data))
            GeoParquet.write(target_file, df)
        end
        return nothing
    end
end

write_storage_variable!(
    storage::GeoStorage,
    group::AbstractString,
    var_name::AbstractString,
    data;
    kwargs...
) = write_storage_variable!(
    storage,
    isempty(group) ? String(var_name) : joinpath(String(group), String(var_name)),
    data;
    kwargs...
)

"""
    _storage_zarr_group(storage, group_path) -> Union{Zarr.ZGroup, Nothing}

Walk to `group_path` from the store's root, creating nothing, and return the group there, or
`nothing` when any component of the path is absent. The two listing functions share it because
the walk is one decision spelled two ways otherwise, and `create_storage_group` and
`has_storage_group` each carry their own copy of the same steps.
"""
function _storage_zarr_group(storage::GeoStorage, group_path::AbstractString)
    cur = storage.handle
    for p in split(replace(group_path, "\\" => "/"), "/")
        s = String(p)
        isempty(s) && continue
        haskey(cur, s) || return nothing
        cur = cur[s]
    end
    return cur
end

"""
    list_storage_groups(storage::GeoStorage; group_path::AbstractString = "") -> Vector{String}

The group names one level below `group_path`, sorted. An empty `group_path` asks about the
store's root.

Both functions here and `list_storage_variables` were exported before they existed: importing
either raised `UndefVarError`, and nothing caught it because no test and no caller used them.
A store that cannot be listed cannot be inspected after the fact, which is the whole point of
the grouped layout.
"""
function list_storage_groups(storage::GeoStorage; group_path::AbstractString = "")
    if storage.backend == :zarr
        grp = _storage_zarr_group(storage, group_path)
        grp === nothing && return String[]      # a path that is not there has no groups
        return sort(String.(collect(keys(grp.groups))))
    elseif storage.backend == :geoparquet
        # A group is a directory under the storage root; a nested path is relative to it.
        base = isempty(group_path) ? dirname(storage.uri) :
               joinpath(dirname(storage.uri), group_path)
        isdir(base) || return String[]
        return sort([String(d) for d in readdir(base) if isdir(joinpath(base, d))])
    end
    return String[]
end

"""
    list_storage_variables(storage::GeoStorage; group::AbstractString = "") -> Vector{String}

The variable names written directly inside `group`, sorted, without the path prefix. An empty
`group` asks about the store's root. A variable in a nested group is listed by naming that group.
"""
function list_storage_variables(storage::GeoStorage; group::AbstractString = "")
    if storage.backend == :zarr
        grp = _storage_zarr_group(storage, group)
        grp === nothing && return String[]      # a path that is not there has no variables
        return sort(String.(collect(keys(grp.arrays))))
    elseif storage.backend == :geoparquet
        base = isempty(group) ? dirname(storage.uri) :
               joinpath(dirname(storage.uri), group)
        isdir(base) || return String[]
        return sort([String(replace(f, ".parquet" => "")) for f in readdir(base)
                     if f isa String && endswith(String(f), ".parquet")])
    end
    return String[]
end

"""
    read_storage_variable(storage::GeoStorage, var_path::AbstractString)

Read an array or tabular dataset from `storage` at `var_path`.
"""
function read_storage_variable(storage::GeoStorage, var_path::AbstractString)
    if storage.backend == :zarr
        parts = split(replace(var_path, "\\" => "/"), "/")
        cur = storage.handle
        for p in parts
            s = String(p)
            isempty(s) && continue
            cur = cur[s]
        end
        # `cur[:]` flattens a multi-dimensional array to a vector. Index by range per axis
        # instead: a 7x5 field read back as 35 numbers has lost the shape that was written,
        # and the caller cannot tell that from the values.
        return cur[ntuple(i -> 1:size(cur, i), ndims(cur))...]
    elseif storage.backend == :geoparquet
        target_file = if endswith(var_path, ".parquet")
            joinpath(dirname(storage.uri), var_path)
        else
            joinpath(dirname(storage.uri), var_path * ".parquet")
        end
        isfile(target_file) || error("Variable file not found at $(target_file)")
        return GeoParquet.read(target_file)
    end
end

read_storage_variable(
    storage::GeoStorage,
    group::AbstractString,
    var_name::AbstractString
) = read_storage_variable(
    storage,
    isempty(group) ? String(var_name) : joinpath(String(group), String(var_name))
)

"""
    lakehouse_dir(subpaths...; create::Bool=false) -> String

Resolve the canonical root directory of the shared data lakehouse across all
projects. The resolution order is:
1. Environment variable `TETHYS_LAKEHOUSE` or `GEODATA_LAKEHOUSE` (highest precedence).
2. Project-level preference in `LocalPreferences.toml` (`[Tethys] lakehouse_dir`).
3. User-level configuration file `~/.tethys/storage.toml` or `~/.geodata/storage.toml` (`[storage] store_root`).
4. Workspace proximity fallback `normpath(joinpath(@__DIR__, "..", "..", "..", "lakehouse"))`.

If `subpaths` are provided, they are concatenated to the resolved root using `joinpath`.
When `create=true`, any missing intermediate directories are created on disk.
"""
function lakehouse_dir(subpaths...; create::Bool = false)::String
    root = if haskey(ENV, "TETHYS_LAKEHOUSE") && !isempty(ENV["TETHYS_LAKEHOUSE"])
        ENV["TETHYS_LAKEHOUSE"]
    elseif haskey(ENV, "GEODATA_LAKEHOUSE") && !isempty(ENV["GEODATA_LAKEHOUSE"])
        ENV["GEODATA_LAKEHOUSE"]
    else
        local_pref = joinpath(pwd(), "LocalPreferences.toml")
        val = ""
        if isfile(local_pref)
            try
                parsed = TOML.parsefile(local_pref)
                if haskey(parsed, "Tethys") && haskey(parsed["Tethys"], "lakehouse_dir")
                    val = String(parsed["Tethys"]["lakehouse_dir"])
                end
            catch
                val = ""
            end
        end
        if isempty(val)
            user_cfg_tethys = joinpath(homedir(), ".tethys", "storage.toml")
            user_cfg_geo = joinpath(homedir(), ".geodata", "storage.toml")
            user_cfg = isfile(user_cfg_tethys) ? user_cfg_tethys : (isfile(user_cfg_geo) ? user_cfg_geo : "")
            if !isempty(user_cfg)
                try
                    parsed = TOML.parsefile(user_cfg)
                    haskey(parsed, "storage") && haskey(parsed["storage"], "store_root") ?
                        String(parsed["storage"]["store_root"]) : ""
                catch
                    ""
                end
            else
                ""
            end
        else
            val
        end
    end

    if isempty(root)
        candidate = normpath(joinpath(@__DIR__, "..", "..", "..", "lakehouse"))
        root = candidate
    end

    target = isempty(subpaths) ? root : joinpath(root, subpaths...)
    if create && !isdir(target)
        mkpath(target)
    end
    return normpath(target)
end

"""
    set_lakehouse_dir!(path::AbstractString)

Persistently set the shared lakehouse root path in `LocalPreferences.toml` under
`[Tethys] lakehouse_dir`.
"""
function set_lakehouse_dir!(path::AbstractString)
    local_pref = joinpath(pwd(), "LocalPreferences.toml")
    cfg = isfile(local_pref) ? (try TOML.parsefile(local_pref) catch; Dict{String, Any}() end) : Dict{String, Any}()
    tethys_dict = get!(cfg, "Tethys", Dict{String, Any}())
    tethys_dict["lakehouse_dir"] = abspath(path)
    open(local_pref, "w") do io
        TOML.print(io, cfg)
    end
    return abspath(path)
end

"""
    project_lakehouse_path(path::AbstractString, project_root::AbstractString) -> String

Resolve a configured output path against a *project root* rather than the current working
directory. All lakehouse references resolve canonically to the shared sibling workspace
lakehouse:

- `../lakehouse/...` -> resolves relative to `project_root` into the sibling `../lakehouse`
- `lakehouse/...`    -> resolved to the shared sibling `../lakehouse/...`

An absolute path is returned unchanged. A relative path that does not name a lakehouse
(e.g. `outputs/run`) is anchored to `project_root`.

# Inputs
- `path`: The configured path, relative or absolute.
- `project_root`: The root of the project the path belongs to.

# Outputs
An absolute normalized path.
"""
function project_lakehouse_path(path::AbstractString, project_root::AbstractString)::String
    isabspath(path) && return normpath(path)

    root = normpath(project_root)
    cleaned = normpath(path)

    # Lakehouse paths always resolve to the shared sibling workspace lakehouse:
    # `lakehouse/...` or `../lakehouse/...` -> `projects/lakehouse/...`
    if startswith(cleaned, "lakehouse")
        return normpath(joinpath(root, "..", cleaned))
    end

    return normpath(joinpath(root, path))
end

