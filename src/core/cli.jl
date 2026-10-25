"""
    cli.jl

Standardized command-line argument parsing and execution dispatch across projects.
"""
module CLI

export parse_cli_args, print_cli_help

"""
    parse_cli_args(args::Vector{String}; allowed_flags = String[], default_config = "") -> Dict{String, Any}

Parse standard CLI options into a dictionary:
- `--config=path`
- `--output-dir=path`
- `--overwrite`
- `--help`, `-h`
- Custom flags and segment selectors (e.g. `--all`, `--grid`, `--sim`, `--track`)
"""
function parse_cli_args(
    args::Vector{String};
    allowed_flags::Vector{String} = String[],
    default_config::AbstractString = ""
)::Dict{String, Any}
    opts = Dict{String, Any}(
        "config" => default_config,
        "output_dir" => "",
        "overwrite" => false,
        "help" => false,
        "actions" => Symbol[]
    )

    for a in args
        s = strip(a)
        if s in ("--help", "-h")
            opts["help"] = true
        elseif startswith(s, "--config=")
            opts["config"] = String(split(s, '=', limit=2)[2])
        elseif startswith(s, "--output-dir=")
            opts["output_dir"] = String(split(s, '=', limit=2)[2])
        elseif s == "--overwrite"
            opts["overwrite"] = true
        elseif startswith(s, "--")
            flag = s[3:end]
            val = true
            if occursin("=", flag)
                parts = split(flag, '=', limit=2)
                flag = parts[1]
                val = parts[2]
            end
            opts[flag] = val
            push!(opts["actions"], Symbol(flag))
        else
            # Positional action/segment argument
            push!(opts["actions"], Symbol(lowercase(s)))
        end
    end

    return opts
end

"""
    print_cli_help(io::IO, project_name::AbstractString, desc::AbstractString, flags::Dict{String, String})

Standard formatted CLI help printer.
"""
function print_cli_help(
    io::IO,
    project_name::AbstractString,
    desc::AbstractString,
    flags::Dict{String, String}
)
    println(io, "================================================================================")
    println(io, "  $(project_name) -- $(desc)")
    println(io, "================================================================================")
    println(io, "\nUsage:")
    println(io, "  julia --project=. $(project_name)Run.jl [flags]\n")
    println(io, "Options:")
    println(io, "  --config=FILE       Path to configuration TOML file.")
    println(io, "  --output-dir=DIR    Override default output destination.")
    println(io, "  --overwrite         Overwrite previous simulation runs and results.")
    println(io, "  --help, -h          Display this help message and exit.\n")
    if !isempty(flags)
        println(io, "Project Specific Flags:")
        for (k, v) in sort(collect(flags))
            println(io, "  --$(rpad(k, 18)) $(v)")
        end
    end
    println(io, "================================================================================")
end

end # module CLI
