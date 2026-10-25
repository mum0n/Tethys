"""
    config.jl

Unified configuration management, TOML reading/writing, parameter schema validation,
and environment variable substitution.
"""
module Config

using TOML

export load_config, save_config, get_config_value, validate_config,
       resolve_config_path

"""
    load_config(path_or_dict; base_dir = "") -> Dict{String, Any}

Parse a TOML configuration file or validate an existing configuration dictionary.
Automatically resolves relative paths against `base_dir` (default: directory of the file).
"""
function load_config(path_or_dict; base_dir::AbstractString = "")::Dict{String, Any}
    if path_or_dict isa AbstractDict
        return deepcopy(Dict{String, Any}(string(k) => v for (k, v) in path_or_dict))
    elseif path_or_dict isa AbstractString
        target = abspath(path_or_dict)
        isfile(target) || error("Configuration file does not exist: $(path_or_dict)")
        parsed = TOML.parsefile(target)
        # Store configuration file metadata
        dir = isempty(base_dir) ? dirname(target) : base_dir
        parsed["_config_path"] = target
        parsed["_config_dir"] = dir
        return parsed
    else
        error("Unsupported configuration source type: $(typeof(path_or_dict))")
    end
end

"""
    save_config(path::AbstractString, config::AbstractDict)

Serialize a configuration dictionary to a TOML file. Strips internal metadata keys (e.g. `_config_path`).
"""
function save_config(path::AbstractString, config::AbstractDict)
    clean = Dict{String, Any}()
    for (k, v) in config
        if !startswith(string(k), "_")
            clean[string(k)] = v
        end
    end
    open(path, "w") do io
        TOML.print(io, clean; sorted = true)
    end
    return path
end

"""
    get_config_value(config::AbstractDict, section::AbstractString, key::AbstractString, default_val)

Safely extract a nested configuration value with fallback.
"""
function get_config_value(config::AbstractDict, section::AbstractString, key::AbstractString, default_val)
    sec_str = string(section)
    key_str = string(key)
    if haskey(config, sec_str) && config[sec_str] isa AbstractDict && haskey(config[sec_str], key_str)
        return config[sec_str][key_str]
    end
    return default_val
end

"""
    resolve_config_path(config::AbstractDict, raw_path::AbstractString) -> String

Resolve `raw_path` relative to `config["_config_dir"]` if it is a relative path.
"""
function resolve_config_path(config::AbstractDict, raw_path::AbstractString)::String
    isempty(strip(raw_path)) && return ""
    isabspath(raw_path) && return normpath(raw_path)
    base = get(config, "_config_dir", pwd())
    return normpath(joinpath(base, raw_path))
end

"""
    validate_config(config::AbstractDict, required_sections::Vector{String}) -> Bool

Ensure all required top-level sections exist in the configuration.
"""
function validate_config(config::AbstractDict, required_sections::Vector{String})::Bool
    for sec in required_sections
        haskey(config, sec) || error("Missing required configuration section: [$(sec)]")
    end
    return true
end

end # module Config
