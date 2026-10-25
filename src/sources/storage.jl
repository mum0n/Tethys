# ==============================================================================
#  storage.jl -- where assimilated data goes, and it is the user's to change
# ==============================================================================
#
#  Every product GeoDataSources assimilates -- bathymetry, coastline, winds, WOA23,
#  boundary physics, tides -- is kept in ONE central, shared registry rather than in
#  each run's private `outputs/<scenario>/inputs/`. A dataset is fetched once per
#  machine and reused across scenarios, regions and halves, which is the whole point
#  of a lakehouse.
#
#  The location is not a constant buried in code. It is read at call time from
#  `default_storage.toml` beside this package (`GeoDataSources/default_storage.toml`),
#  so an operator relocates the registry by editing one file -- no recompile, no
#  environment variables. Read at call time (not baked into the precompiled image) is
#  what makes a user edit take effect on the next assimilation.
# ==============================================================================

"""
    storage_config() -> Dict{String, Any}

The parsed assimilation-storage configuration, i.e. `default_storage.toml` beside this
package. Returns an empty dict if the file is missing or unparseable, so the built-in
defaults in `store_root()`/`read_config_bool` apply rather than a hard failure.
"""
function storage_config()::Dict{String, Any}
    path = joinpath(dirname(@__DIR__), "default_storage.toml")
    isfile(path) || return Dict{String, Any}()
    try
        parsed = TOML.parsefile(path)
        return haskey(parsed, "storage") ? parsed["storage"]::Dict{String, Any} : Dict{String, Any}()
    catch
        return Dict{String, Any}()
    end
end

"""
    storage_file_path() -> String

Path to the user-editable stub config, so the error message can name it.
"""
storage_file_path() = joinpath(dirname(@__DIR__), "default_storage.toml")

"""
    store_root() -> String

The central registry every assimilated product lives under. Configured by
`default_storage.toml [storage] store_root`, resolved relative to the GeoDataSources
package directory (so a relative value is stable regardless of the caller's working
directory). Defaults to `../lakehouse/Tethys`, which from this package is the
workspace's `projects/lakehouse/Tethys`.
"""
function store_root()::String
    cfg = storage_config()
    if haskey(cfg, "store_root") && !isempty(String(cfg["store_root"]))
        root = String(cfg["store_root"])
        return isabspath(root) ? root : abspath(dirname(@__DIR__), root)
    end
    return lakehouse_dir("Tethys")
end

"""
    store_path(source::AbstractString) -> String

The directory for one source's assimilated products inside the registry (e.g.
`store_path("etopo")`, `store_path("woa23")`).
"""
store_path(source::AbstractString) = joinpath(store_root(), source)

"""
    storage_option(name::AbstractString, default::Bool) -> Bool

A boolean assimilation option from `default_storage.toml`, with a default when unset
(e.g. `checksum.verify`). Returns `default` if the key is missing.
"""
function storage_option(name::AbstractString, default::Bool)::Bool
    cfg = storage_config()
    haskey(cfg, name) ? Bool(cfg[name]) : default
end

"""
    ensure_store_dir(sub::AbstractString) -> String

Make `store_path(sub)` exist and return it, so a fetcher writes into a directory that
is already there.
"""
function ensure_store_dir(sub::AbstractString)::String
    d = store_path(sub)
    mkpath(d)
    return d
end
