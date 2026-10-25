"""
    run_state.jl

What a run leaves behind, and the two ways a run may be asked to clear it.

A run's output directory holds two different kinds of thing, and the difference is the
whole point of the two clearing modes:

  * the MODEL RESULTS -- the store the run writes, the checkpoints it resumes from, the
    trajectories it produced. These are computed. Overwriting them is what "run it again
    from scratch" means, and it costs nothing but the run itself.
  * the DOWNLOADED STORE -- the ingested inputs: the bathymetry, the coastline, the winds,
    the vertical grid, the coastal polygons. These are fetched, not computed, so clearing
    them costs a download, and sometimes a data account to download them with.

`--overwrite` clears the first and keeps the second. `--blank_state` clears both.

Nothing in here knows which paths belong to which kind of state: the two run drivers name
their own, and this file only decides what a purge does with the names it is given. That
is deliberate. The shared Tethys store under the lakehouse holds datasets that several
runs depend on, and a purge that decided for itself which paths were "the downloaded
store" would eventually reach outside a run's own directory and take a neighbour with it.
"""

export RunStateTarget,
       run_state_targets,
       confirm_run_state_purge,
       apply_run_state_purge,
       purge_run_state

"""
    RunStateTarget(path, kind, label)

One path a state purge would remove, and what that path is.

`kind` is `:results` for the model outputs -- the store the run writes, the checkpoints it
resumes from, the trajectories it produced -- and `:inputs` for the downloaded store: the
ingested bathymetry, coastline, winds and vertical grid.
"""
struct RunStateTarget
    path::String
    kind::Symbol
    label::String
end

"""
    _expand_pattern(pattern) -> Vector{String}

The existing paths a single-level pattern matches.

The pattern names a directory and a file name pattern, separated by the last path
separator: `checkpoints/*.jld2` is the `*.jld2` files in `checkpoints`, beside the
pattern's own directory. `*` stands for any run of characters; everything else is
literal. The listing is not recursive, on purpose: a purge that reached into
subdirectories would remove things the pattern did not name.
"""
function _expand_pattern(pattern::AbstractString)
    p = String(pattern)
    dir, name = if occursin('/', p) || occursin('\\', p)
        parts = split(p, r"[/\\]")
        join(parts[1:(end - 1)], "/"), parts[end]
    else
        ".", p
    end
    isdir(dir) || return String[]
    # Escape everything that is not a wildcard, then let `*` become "any run of characters".
    escaped = replace(name, r"([\\^$.|?+()\[\]{}])" => s"\\\1")
    rx = Regex("^" * replace(escaped, "*" => ".*") * "\$")
    return [joinpath(dir, f) for f in sort(readdir(dir)) if occursin(rx, f)]
end

"""
    run_state_targets(output_dir; results, result_globs, inputs, blank_state) -> Vector{RunStateTarget}

The paths a purge would remove, and only those that are actually there.

`results` are the model-result paths, absolute or relative to `output_dir`. `result_globs`
are glob patterns for model results whose names are not fixed -- the checkpoints, whose
file names carry the iteration they were written at. Both may be absolute; a relative one
is taken against `output_dir`.

`inputs` are the downloaded store's paths. They are returned only when `blank_state` is
true: overwriting the model results keeps the downloaded store, because the point of
keeping it is that the next run does not have to fetch it again.
"""
function run_state_targets(output_dir::AbstractString;
                           results::AbstractVector{<:AbstractString} = String[],
                           result_globs::AbstractVector{<:AbstractString} = String[],
                           inputs::AbstractVector{<:AbstractString} = String[],
                           blank_state::Bool = false)
    base = abspath(String(output_dir))
    out = RunStateTarget[]
    seen = Set{String}()

    function add!(path, kind, label)
        p = normpath(isabspath(path) ? String(path) : joinpath(base, String(path)))
        (p in seen) && return nothing
        ispath(p) || return nothing
        push!(seen, p)
        push!(out, RunStateTarget(p, kind, label))
        return nothing
    end

    for r in results
        add!(r, :results, "model results")
    end
    for g in result_globs
        pattern = isabspath(g) ? String(g) : joinpath(base, String(g))
        for m in _expand_pattern(pattern)
            add!(m, :results, "model results")
        end
    end
    if blank_state
        for i in inputs
            add!(i, :inputs, "downloaded store")
        end
    end
    return out
end

"""
    _warn_run_state_purge(targets; blank_state = false, io = stdout)

Print what a purge would delete, without asking.

The warning is printed whether or not the question is asked, so an operator can see what
went either way. Split out from [`confirm_run_state_purge`](@ref) so that a caller who
has already been told and has answered still gets the list.
"""
function _warn_run_state_purge(targets::AbstractVector{RunStateTarget};
                               blank_state::Bool = false, io::IO = stdout)
    results = [t for t in targets if t.kind === :results]
    inputs = [t for t in targets if t.kind === :inputs]

    println(io, "="^73)
    println(io, " WARNING -- this run will DELETE existing state on disk.")
    println(io, "="^73)
    if !isempty(results)
        println(io, " Model results to be removed:")
        for t in results
            println(io, "   - ", t.path)
        end
    end
    if !isempty(inputs)
        println(io, " Downloaded store to be removed (it will be fetched again):")
        for t in inputs
            println(io, "   - ", t.path)
        end
    end
    if blank_state
        println(io, " The downloaded store is gone: the next run fetches it again.")
    else
        println(io, " The downloaded store is KEPT: the next run reuses it and does not")
        println(io, " fetch it again. The shared Tethys store is never touched by this.")
    end
    return nothing
end

"""
    confirm_run_state_purge(targets; blank_state = false, io = stdout, input = stdin) -> Bool

Warn about what a purge would delete and ask for a yes or a no.

The answer defaults to no: an empty line, anything that is not `y` or `yes`, and an
unreadable input all mean "do not delete". A purge that cannot ask -- because the run is
not attached to a terminal -- is refused rather than assumed. There is no flag that
answers the question in advance: whoever asked for a deletion is the one who has to
answer for it. A caller that has already been told and has answered should not come
here at all; it should call [`purge_run_state`](@ref) with `confirmed = true`.

`input` is the stream the answer is read from; it is `stdin` in a real run, and a buffer
in a test.
"""
function confirm_run_state_purge(targets::AbstractVector{RunStateTarget};
                                 blank_state::Bool = false,
                                 io::IO = stdout,
                                 input::IO = stdin)
    _warn_run_state_purge(targets; blank_state, io)

    if input === stdin && !isa(stdin, Base.TTY)
        println(io, "")
        println(io, " Not proceeding: there is no terminal to ask on, and a deletion")
        println(io, " that nobody answered for is not one this run may make.")
        return false
    end

    print(io, " Proceed? [y/N]: ")
    flush(io)
    answer = try
        readline(input)
    catch
        ""
    end
    if lowercase(strip(String(answer))) in ("y", "yes")
        return true
    end
    println(io, " Not proceeding. Nothing was removed.")
    return false
end

"""
    apply_run_state_purge(targets; io = stdout) -> Vector{String}

Remove the targets and report what was removed.
"""
function apply_run_state_purge(targets::AbstractVector{RunStateTarget}; io::IO = stdout)
    removed = String[]
    for t in targets
        ispath(t.path) || continue
        try
            rm(t.path; recursive = true, force = true)
            push!(removed, t.path)
            println(io, " Removed: ", t.path)
        catch err
            @error "Could not remove a run-state path" path = t.path exception = err
        end
    end
    return removed
end

"""
    purge_run_state(output_dir; results, result_globs, inputs, blank_state,
                    io, input, confirmed = false) -> Bool

Enumerate, warn, ask, and remove. Returns `true` when the run may proceed and `false` when
the operator declined -- in which case the caller must not run, and nothing was removed.

`confirmed = true` means the caller has already been told what will be deleted and has
said yes. The warning is still printed -- the operator should see what went -- but the
question is not asked again.

That distinction matters because a purge with no way to ask is refused, not assumed. A
caller that CAN ask should ask. A caller that has already been told and has answered --
a command-line flag, a scheduled job, an agent running a documented command -- has
already been told and has answered, and asking again makes the flag unusable in exactly
the situation it exists for.
"""
function purge_run_state(output_dir::AbstractString;
                         results::AbstractVector{<:AbstractString} = String[],
                         result_globs::AbstractVector{<:AbstractString} = String[],
                         inputs::AbstractVector{<:AbstractString} = String[],
                         blank_state::Bool = false,
                         io::IO = stdout,
                         input::IO = stdin,
                         confirmed::Bool = false)
    targets = run_state_targets(output_dir; results, result_globs, inputs, blank_state)
    if isempty(targets)
        println(io, "No existing state to clear in ", abspath(String(output_dir)), ".")
        return true
    end
    if !confirmed
        confirm_run_state_purge(targets; blank_state, io, input) || return false
    else
        _warn_run_state_purge(targets; blank_state, io)
    end
    apply_run_state_purge(targets; io)
    return true
end
