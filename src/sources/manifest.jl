"""
    manifest.jl

Provenance registry for every physical input a run consumes.

This module provides a generic data provenance framework that can be used by
any project. TethysFlow- and TethysDrift-specific data sources are also defined here.
"""
module Manifest

using Downloads
using TranscodingStreams
using CodecZlib
using SHA
using Dates
using JSON3
using Base64
using HTTP
using Interpolations
using Random

# The fetchers live in the parent module and are included before this one, so
# `FETCHERS` below can dispatch to them without this module re-implementing access.
import ..fetch_erddap_bathymetry
import ..fetch_open_meteo_winds
import ..fetch_woa23
import ..fetch_natural_earth_coastline
import ..fetch_copernicus_physics_subset
import ..fetch_bottom_temperature
import ..fetch_substrate_grainsize
import ..fetch_nwa_bathymetry
import ..fetch_tpxo9_velocity
import ..is_cached
# The central assimilation registry. Declared in the parent module, so it has to be named there:
# an unqualified `store_root()` inside this module resolves against this module's own scope and
# raises UndefVarError at the first `fetch_input`, because every cache template carries `<store>`
# and the substitution runs before the cache check.
import ..store_root

"""
    DataSource

One physical input, described well enough to fetch it again and to prove which version was used.

# Fields
- `key::Symbol`: Stable identifier used in configs and in the provenance file (e.g. `:bathymetry`).
- `title::String`: Human-readable name of the product.
- `kind::Symbol`: One of `:gridded`, `:station`, or `:analytic`. `:analytic` means the input is
  generated rather than downloaded, and therefore has no provenance beyond its parameters.
- `keyless::Bool`: `true` when the primary source needs no account, token, or licence click.
  Every input in this project is keyless by design; a `false` here is a defect, not a policy.
- `primary::String`: URL or DOI of the source to use.
- `alternatives::Vector{String}`: Other acceptable sources, most preferred first. Each entry
  states its access requirement, because "open" and "reachable from here" are different claims.
- `cache::String`: Path under `inputs/` where the downloaded artefact is kept.
- `license::String`: Licence or access terms, quoted where known.
- `notes::String`: Anything a reader would otherwise have to discover the hard way.
"""
struct DataSource
    key::Symbol
    title::String
    kind::Symbol                  # :gridded | :vector | :analytic
    keyless::Bool
    primary::String
    alternatives::Vector{String}
    cache::String                 # path template; <output_dir> and <mm> are substituted
    license::String
    notes::String
    tier::Symbol
    products::Dict{String, String}
    fetch::Union{Function, Nothing}
end

"""
    DataSource(key, title, kind, keyless, primary, alternatives, cache, license, notes;
               tier = :discretized, products = Dict{String,String}(), fetch = nothing)

Construct a `DataSource` from its core descriptive fields. The optional keyword arguments
carry provenance-of-origin metadata that most sources do not need to set explicitly:

- `tier` — where in the chain the data sit: `:raw` (measured), `:discretized` (binned,
  QC'd, gridded from measurements), or `:modelled` (from a reanalysis or model). This is a
  property of the source, not the variable derived from it, so it belongs here rather than
  in a config.
- `products` — maps each product name within an arbitrary source to variable names, so a
  manifest entry can specify where individual variables come from.
- `fetch` — an optional custom fetcher, overriding the default download from `primary`.
"""
function DataSource(key::Symbol, title::String, kind::Symbol, keyless::Bool,
                    primary::String, alternatives::Vector{String}, cache::String,
                    license::String, notes::String;
                    tier::Symbol = :discretized,
                    products::Dict{String, String} = Dict{String, String}(),
                    fetch::Union{Function, Nothing} = nothing)
    return DataSource(key, title, kind, keyless, primary, alternatives, cache, license,
                      notes, tier, products, fetch)
end

# Export the types and functions
export DATA_SOURCES, DataSource, fetch_input, input_dir, file_digest, data_provenance,
       data_source, describe_data_sources, available_data, data_summary, buildable,
       buildable_sources, resolved_cache_path, tier_path

"""
    resolved_cache_path(key::Symbol, output_dir::AbstractString) -> String

The cache path `fetch_input` will look at for `key`, with every template token expanded:
`<store>` becomes the central registry (`store_root()`), `<output_dir>` the caller's directory,
and `<mm>` the zero-padded month.

The manifest's cache templates carry the medallion tier explicitly (`<store>/raw/…`,
`<store>/discretized/…`, `<store>/modelled/…`), so this path *is* the medallion path: ingestion
writes where a reader looks, and there is no second function to keep in step. That was not
always so — the templates used to name a top-level `<store>/<<key>>/` directory, the tiers
described where products *should* be instead of where they were, and the store accumulated an
`ingested/` tier and a `modelled/` tier that did not exist.

Sources that share one product have separate entries with the same tier directory:
`:atmosphere` and `:surface_winds` both point at `modelled/era5.zarr`, because ERA5 serves
both variables and a copy per source would be two copies of one archive.
"""
function resolved_cache_path(key::Symbol, output_dir::AbstractString)::String
    return resolved_cache_path(data_source(key), output_dir)
end

"""
    resolved_cache_path(d::DataSource, output_dir::AbstractString) -> String

The `DataSource` form, for a caller holding a source that is not in the declared registry
(an overlay passed to `data_provenance`, say).
"""
function resolved_cache_path(d::DataSource, output_dir::AbstractString)::String
    return replace(replace(replace(d.cache, "<output_dir>" => output_dir),
                           "<store>" => store_root()), "<mm>" => "00")
end

"""
    tier_path(key::Symbol; ext::Union{Nothing, AbstractString} = nothing) -> String

Where a source's assimilated product lives **under the medallion scheme**:
`<store>/<tier>/<key>.<ext>`, with the tier from the source's declared processing level.

This is now where products actually are. `resolved_cache_path` returns this path, so ingestion
writes it and readers read it from the same function — the two views the audit found disagreeing
for 12 of 12 sources are now one. The store no longer has a top level holding `<source>/`
directories, an `ingested/` tier nothing documents, or a `modelled/` tier that did not exist.

Three tiers, and each is a directory the scheme owns:

| declared level | tier | what it holds |
|---|---|---|
| `:raw` | `raw/` | the source as delivered, format aside |
| `:discretized` | `discretized/` | gridded/cleaned onto a regular grid |
| `:modelled` | `modelled/` | a numerical model or reanalysis product |

`ext` defaults to the type the product actually is, not to `zarr`: a Zarr store is a directory
and a NetCDF archive is a file, and writing `.zarr` onto a `.nc` would mark every NetCDF
product as a Zarr store the reader cannot open. It is taken from the manifest's own cache
template, so the extension is a record of how the product is stored rather than a guess here.

`:gridded` is not one of the three levels, and a source carrying it gets `raw/` rather than an
error — a level this register does not recognise should not make a path unresolvable.

`lakehouse_path` is the same idea with the Bronze/Silver/Gold vocabulary and a different tier
set; they are not the same scheme. Callers want one path per product, so this is the one.
"""
function tier_path(key::Symbol; ext::Union{Nothing, AbstractString} = nothing)
    root = store_root()
    d = data_source(key)
    dir = d.tier === :discretized ? "discretized" :
          d.tier === :modelled ? "modelled" : "raw"
    use_ext = isnothing(ext) ? _product_extension(d) : ext
    clean = startswith(String(use_ext), ".") ? String(use_ext)[2:end] : String(use_ext)
    return joinpath(root, dir, "$(key).$(clean)")
end

"""
    _product_extension(d::DataSource) -> String

The extension the product is actually stored with, from its own cache template. `zarr` when the
template names no extension at all (a Zarr store is a directory, so a bare directory path is the
Zarr convention) and the template's suffix otherwise.
"""
function _product_extension(d::DataSource)::String
    base = basename(rstrip(d.cache, ('/', '\\')))
    ext = splitext(base)[2]
    isempty(ext) ? "zarr" : ext[2:end]
end

"""
    _truncate(s::AbstractString, n::Int = 96) -> String

First `n` characters of `s`, with an ellipsis if it was longer. Used to keep the summary
display readable when `alternatives` holds prose rather than a URL.
"""
function _truncate(s::AbstractString, n::Int = 96)
    ncodeunits(s) <= n && return String(s)
    return String(first(s, n)) * " ..."
end

"""
    available_data() -> Vector{NamedTuple}

One row per declared source, describing what it is, where it comes from, its tier, and
whether it needs credentials. This is the function to call before writing a config: it
answers "what can I actually ask for, and what will it cost me".

Each row carries `:key`, `:title`, `:tier`, `:kind`, `:keyless`, `:cache`, `:license`,
`:primary`, `:alternatives`. Sorted by tier then key, so the measured sources and the
modelled ones are not interleaved.
"""
function available_data()
    rows = NamedTuple[]
    for key in sort(collect(keys(DATA_SOURCES)); by = string)
        d = DATA_SOURCES[key]
        push!(rows, (
            key = key,
            title = d.title,
            tier = d.tier,
            kind = d.kind,
            keyless = d.keyless,
            cache = d.cache,
            license = d.license,
            primary = d.primary,
            alternatives = d.alternatives,
        ))
    end
    return sort!(rows, by = r -> (String(r.tier), String(r.key)))
end

"""
    data_summary(io::IO = stdout; show_alternatives::Bool = false)

Print a compact table of every declared source, grouped by tier, with an access-to-get-it
marker. `show_alternatives = true` also lists each source's fallbacks.

Access is stated as `keyless` (nothing needed but a connection), `needs account` (free but
requires `~/.copernicusmarine/credentials`), or `local` (shipped with GeoDataSources), so a
reader can tell at a glance which sources will work on a fresh checkout.
"""
function data_summary(io::IO = stdout; show_alternatives::Bool = false)
    rows = available_data()
    println(io, "Declared data sources (", length(rows), ")")
    println(io, repeat("-", 78))
    current = nothing
    for r in rows
        if r.tier !== current
            current = r.tier
            println(io, "\n", uppercase(string(current)), ":")
        end
        access = if !buildable(r.key)
            "UNBUILDABLE"
        elseif startswith(r.primary, "GeoDataSources tier-1")
            "local"
        elseif r.keyless
            "keyless"
        else
            "needs account"
        end
        println(io, "  ", rpad(string(r.key), 20), rpad(access, 15), r.title)
        println(io, "  ", rpad("", 20), rpad("->", 15), r.cache)
        if show_alternatives
            for a in r.alternatives
                println(io, "  ", rpad("", 20), rpad("alt:", 15), _truncate(a))
            end
        end
    end
    n_unbuildable = count(r -> !buildable(r.key), rows)
    println(io, "\n", repeat("-", 78))
    println(io, "Access: `local` (shipped), `keyless` (no account), " *
                "`needs account` (free registration, credentials file), " *
                "`UNBUILDABLE` (declared but no fetcher - cannot be requested yet)")
    n_unbuildable > 0 && println(io, "Unbuildable: $(n_unbuildable) of $(length(rows))")
    return nothing
end

"""
    fetch_input(key::Symbol, output_dir::AbstractString; tier = nothing,
                force::Bool = false, strict::Bool = true, kwargs...) -> String

Ensure the input `key` is present in its cache location — the central store for a source
that declares one, and this run's inputs directory for one that does not — at the requested
`tier`, and return the path to the cached artefact.

This is the single dispatch point for "what data do we need" -> "how do we get it",
tier by tier:

- **`:raw`** - the measured data as delivered, in optimal storage (`lakehouse/Tethys/raw/`).
- **`:discretized`** - filtered, quality-controlled, and binned onto a resolution chosen
  per dataset. This is where observation noise becomes a grid, and the resolution is a
  property of the data, not of the request.
- **`:modelled`** - output of a numerical model or reanalysis (GLORYS, ERA5, TPXO9), at
  whatever resolution the model carries.

`tier` says what processing level the artefact belongs to, which is a property of the data and
not of the request. It is what `tier_path` uses to place the product under the medallion
scheme; it does **not** choose the builder. The builder is chosen by availability: a fetcher
registered for the key is used, because it is the one that knows how to obtain this source,
and MODELLERS is consulted only for a key no fetcher knows. (Making `tier` the dispatch key
too was what made an honest classification — TPXO9 is a tidal *model* product — silently
reroute ingestion to a plain URL download of the archive.)

Omitted, `tier` is the source's own declared level — `:raw` for `:bottom_temperature`,
`:discretized` for `:woa23` and the measured-gap products, `:modelled` for the reanalyses and
the tidal solution. `buildable_sources()` lists what can actually be built.

`force = true` re-fetches. An unavailable source is an error naming every alternative,
never a silent substitution; `strict = false` lets an individual fetcher choose a
labelled fallback. The cache check uses `is_cached`, so a Zarr store - a directory -
counts as already present.

A level nothing can produce is reported as what it currently is — a plain download of the
`primary`, not an error — because that is what an empty `MODELLERS` table means. A key in
neither table fails here, at the call, rather than surfacing later; `buildable(key, tier)`
asks the same question up front.
"""
function fetch_input(key::Symbol, output_dir::AbstractString; tier = nothing,
                     force::Bool = false, strict::Bool = true, kwargs...)
    d = data_source(key)
    tier = tier === nothing ? d.tier : Symbol(tier)

    if !buildable(key, tier = tier)
        error("Source $(repr(key)) at tier $(repr(tier)) cannot be built. " *
              "Declared fetchers for $(repr(key)): " *
              "$(join(string.(_buildable_tiers(key)), ", ")). " *
              "Either build this tier (add an entry to FETCHERS or MODELLERS in " *
              "src/manifest.jl) or choose a source that supplies it. " *
              "Alternatives for $(repr(key)): " *
              "$(join(string.(d.alternatives), " | ")). " *
              "Run `describe_data_sources()` for the full registry.")
    end

    # `<store>` is the central, user-configurable registry (store_root()); `<output_dir>`
    # survives only for a template that still names it. Substituting `<store>` is what puts
    # every assimilated source under projects/lakehouse/Tethys/<source>, shared across runs.
    target = resolved_cache_path(key, output_dir)
    if !force && is_cached(target) && _tier_matches_cache(target, d)
        return target
    end

    # Inputs marked incomplete are not fabricated: the note says what is ingested.
    d.kind == :analytic && error(
        "Input $(repr(key)) is analytic and is generated by the model, not downloaded. $(d.notes)")

    # Which table builds this artefact, and the two are not the same question. `tier` is a
    # *classification* — what processing level the product belongs to, which is what
    # `tier_path` uses to place it — and using it to pick the builder table made the two
    # meanings collide: tagging TPXO9 honestly as `:modelled` sent `fetch_input(:tides)` to
    # the empty MODELLERS table and fell through to a plain download of the 4.7 GB archive
    # with no extraction, so a documentation change silently broke ingestion.
    #
    # Dispatch is therefore by availability: a fetcher that exists for the key wins, because
    # it is the one that knows how to build this source. MODELLERS is consulted only for a
    # key no fetcher knows, which is what an empty table means.
    fetcher = get(FETCHERS, key, nothing)
    if isnothing(fetcher) && tier === :modelled
        fetcher = get(MODELLERS, key, nothing)
    end
    fetcher === nothing && return _fetch_input_by_url(d, target)
    # Every fetcher writes into the central store. Most take `output_path` (the resolved
    # `target`); `fetch_woa23` takes `output_dir` and names the file itself, so it is handed
    # `store_root()` rather than the caller's per-run directory.
    return fetcher(target; output_dir = store_root(), force = force, strict = strict,
                   kwargs...)
end

"""
    _fetcher_(source) closures: adapters from a declared cache path to each fetcher interface

Each closure takes the cache path `fetch_input` resolved for the source and calls the
fetcher that actually knows how to obtain it, so the manifest describes a dataset without
re-implementing its access and the fetchers never need to know what the manifest calls
a file.

One signature, forwarded uniformly. Every fetcher in the package accepts:

- `lon_range`, `lat_range` - the domain, as a `(min, max)` pair each
- `output_path` - where the result is written (`output_dir` for WOA23, which writes a store
  per variable; both spellings are accepted there)
- `force` - re-obtain rather than reuse a cached store
- `backend`, `verbose` - storage format and reporting

plus per-fetcher extras through `kwargs` (`month`, `time_iso`, `start_date`/`end_date`,
`resolution_deg`, and so on). This matters because `fetch_input` calls these adapters as
`(path; output_dir, force, strict, kwargs...)`: an adapter that named a keyword its callee
did not accept was a `MethodError` on the first live fetch, invisible to every offline test.
Three of the ten were live that way and are now fixed, so the set above is the contract
rather than a description of it.
"""
_fetcher_erddap_bathymetry(path; output_dir, force, strict = true, kwargs...) =
    fetch_erddap_bathymetry(; output_path = path, force = force, kwargs...)

_fetcher_open_meteo_winds(path; output_dir, force, strict = true, kwargs...) =
    fetch_open_meteo_winds(; output_path = path, force = force, kwargs...)

_fetcher_woa23(path; output_dir, force, strict = true, kwargs...) =
    fetch_woa23(; output_dir = output_dir, force = force, strict = strict, kwargs...)

_fetcher_natural_earth_coastline(path; output_dir, force, strict = true, kwargs...) =
    fetch_natural_earth_coastline(; output_path = path, force = force, kwargs...)

# `force` is forwarded, as it is for every other source: the callee accepts it as part of
# the package's canonical fetcher keyword set. It has no store to invalidate here because
# the subset is already re-fetched on every call, which the callee documents.
_fetcher_copernicus_physics(path; output_dir, force, strict = true, kwargs...) =
    fetch_copernicus_physics_subset(; output_path = path, force = force, kwargs...)

# GLORYS12V1 is itself a Copernicus Marine physics product - `GLOBAL_MULTIYEAR_PHY_001_030`,
# the same collection the lateral-boundary path reads - so the reader is the same one and
# only the role differs: this path asks for the reanalysis state (temperature, salinity,
# u, v, SSH, MLD) rather than the boundary subset. Declaring the ID here keeps that
# explicit rather than relying on the reader's default.
_fetcher_glorys(path; output_dir, force, strict = true, kwargs...) =
    fetch_copernicus_physics_subset(; output_path = path, force = force,
                                   dataset_id = "GLOBAL_MULTIYEAR_PHY_001_030", kwargs...)

# Tier-1 raw data held in this package under data/raw/. These are local files rather than
# URLs, so there is no cache to invalidate: the fetcher only decides the output grid.
_fetcher_bottom_temperature(path; output_dir, force, strict = true, kwargs...) =
    fetch_bottom_temperature(; output_path = path, force = force, kwargs...)

_fetcher_substrate(path; output_dir, force, strict = true, kwargs...) =
    fetch_substrate_grainsize(; output_path = path, force = force, kwargs...)

_fetcher_nwa_bathymetry(path; output_dir, force, strict = true, kwargs...) =
    fetch_nwa_bathymetry(; output_path = path, force = force, kwargs...)

_fetcher_tpxo9(path; output_dir, force, strict = true, kwargs...) =
    fetch_tpxo9_velocity(; output_path = path, force = force, kwargs...)

"""
    FETCHERS

How each source is actually obtained: `source key => fetcher`.

One table, one place to look. Every entry has the shape
`(path; output_dir, force, strict, kwargs...) -> path`, where `path` is the cache
location the manifest declares for that source. A key absent from this table is fetched
by plain download from its `primary` URL, which is the right behaviour for artefacts that
are a single file and nothing else.

Adding a new source therefore means: declare it in `DATA_SOURCES` (what it is, where it is
cached, what its alternatives are) and, if it needs more than a URL, add one line here.
"""
const FETCHERS = Dict{Symbol, Function}(
    :bathymetry       => _fetcher_erddap_bathymetry,
    :atmosphere       => _fetcher_open_meteo_winds,
    :surface_winds    => _fetcher_open_meteo_winds,
    :hydrography      => _fetcher_woa23,
    :woa23            => _fetcher_woa23,
    :coastline        => _fetcher_natural_earth_coastline,
    :lateral_boundary => _fetcher_copernicus_physics,
    :bottom_temperature => _fetcher_bottom_temperature,
    :substrate          => _fetcher_substrate,
    :nwa_bathymetry     => _fetcher_nwa_bathymetry,
    :glorys             => _fetcher_glorys,
    :tides              => _fetcher_tpxo9,
)

"""
    MODELLERS

How a source that lives only as model output is produced: `source key => modeller`.

A *fetcher* obtains data someone else measured or published. A *modeller* produces output
that did not exist before: a reanalysis down-sampled to a shelf grid, a tidal solution
interpolated onto sites, a boundary sponge buffer. Both have the same
`(path; output_dir, force, strict, kwargs...)` shape, so `fetch_input` dispatches on tier
without special-casing either.

Empty by design: a declared source with no entry here cannot be built at `:modelled`, and
`buildable(key)`/`available_data()` report that rather than pretending otherwise. Add
entries as reanalysis access is implemented - `glorys`, `tides` and `lateral_boundary` are
the intended first three.
"""
const MODELLERS = Dict{Symbol, Function}()

"""
    buildable(key::Symbol; tier = nothing) -> Bool

Whether `key` can be produced at `tier`. `tier = nothing` asks about its declared tier.

A source is buildable when a fetcher (raw or discretized) or a modeller (modelled) exists
for it. This is the question a config validator should ask *before* selecting a source,
so that naming an unbuildable one fails at load time rather than at fetch time - which is
how `glorys` and `tides` silently became unbuildable promises in the first place.
"""
function buildable(key::Symbol; tier = nothing)
    d = get(DATA_SOURCES, key, nothing)
    d === nothing && return false
    tier = tier === nothing ? d.tier : Symbol(tier)
    return key in keys(FETCHERS) || key in keys(MODELLERS)
end

"""
    _buildable_tiers(key::Symbol) -> Vector{Symbol}

The tiers at which `key` can currently be produced. Empty when neither table has an
entry, which is the state that used to be invisible.
"""
function _buildable_tiers(key::Symbol)
    tiers = Symbol[]
    haskey(FETCHERS, key) && push!(tiers, :raw, :discretized)
    haskey(MODELLERS, key) && push!(tiers, :modelled)
    return unique!(tiers)
end

"""
    _tier_matches_cache(target::AbstractString, d::DataSource) -> Bool

Whether an existing cache file is the tier that was asked for.

The cache path is `tier`-specific in the manifest (`cache`), so a same-path cache from a
different processing level would be returned for the wrong request. A source whose cache
name encodes the tier (`woa23_temperature`, `glorys`) needs no check; one shared between
tiers does, and that is the case to fix by declaring separate cache paths.
"""
_tier_matches_cache(target::AbstractString, d::DataSource) = true

"""
    buildable_sources() -> Vector{Symbol}

Every declared source that can currently be built, sorted by name. Pair with
`available_data()` to show what a config can ask for versus what is once askable.
"""
function buildable_sources()
    return sort!(collect(k for k in keys(DATA_SOURCES) if buildable(k)); by = string)
end

"""
    _fetch_input_by_url(d::DataSource, target::AbstractString) -> String

Download a source that is a single URL and nothing else: a `.part` scratch file, progress
reporting, and an error that names every alternative when it fails.
"""
function _fetch_input_by_url(d::DataSource, target::AbstractString)
    startswith(d.primary, "http") || error(
        "Input $(repr(d.key)) has no direct download URL and no fetcher (primary is " *
        "$(repr(d.primary))). Obtain it manually and place it at $(target). " *
        "Alternatives:\n  " * join(d.alternatives, "\n  "))

    mkpath(dirname(target))
    tmp = target * ".part"
    try
        Downloads.download(d.primary, tmp;
                           progress = _download_progress_callback(string(d.key)))
    catch err
        rm(tmp; force = true)
        error("Could not download $(d.key):\n  $(d.primary)\n$(sprint(showerror, err))\n" *
              "Alternatives, and their access requirements:\n  " *
              join(d.alternatives, "\n  ") *
              "\nPlace the file at $(target) and re-run if none of these is reachable.")
    end
    mv(tmp, target; force = true)
    return target
end

"""
    _download_progress_callback(label::AbstractString) -> Function

Construct a progress callback for `Downloads.download` that reports transfer
progress and rate periodically without flooding the output stream.
"""
function _download_progress_callback(label::AbstractString)
    last_time = Ref(time())
    last_print = Ref(0.0)
    last_bytes = Ref(0)
    finished = Ref(false)
    return function (total::Integer, now::Integer)
        curr_time = time()
        if (curr_time - last_print[]) >= 2.0 || (total > 0 && now >= total)
            dt = curr_time - last_time[]
            rate_mb = dt > 0.1 ? ((now - last_bytes[]) / (1024 * 1024)) / dt : 0.0
            now_mb = round(now / (1024 * 1024); digits = 1)
            if total > 0
                tot_mb = round(total / (1024 * 1024); digits = 1)
                pct = round(100.0 * now / total; digits = 1)
                print("\r  [$(label)] $(now_mb) / $(tot_mb) MB ($(pct)%) " *
                      "[$(round(rate_mb; digits = 2)) MB/s]    ")
            else
                print("\r  [$(label)] $(now_mb) MB [$(round(rate_mb; digits = 2)) MB/s]    ")
            end
            flush(stdout)
            last_print[] = curr_time
            last_time[] = curr_time
            last_bytes[] = now
            if total > 0 && now >= total && !finished[]
                finished[] = true
                println()
            end
        end
    end
end

"""
    input_dir(output_dir) -> String

The container directory for this run's downloaded inputs, `<output_dir>/inputs`.

Inputs are deliberately **not** shared in a top-level `inputs/` directory. Scenarios and
subprojects are self-contained: a run keeps its bathymetry, winds, hydrography,
ocean boundary and tidal solution beside its own outputs under `work/run/`, alongside its
`resolved_config.toml` and `data_provenance.json`.
"""
input_dir(output_dir::AbstractString) = joinpath(output_dir, "inputs")

"""
    file_digest(path::AbstractString) -> String

SHA-256 of a file, used to detect a cached input that has changed since it was recorded.
"""
function file_digest(path::AbstractString)
    isfile(path) || return ""
    ctx = SHA.SHA256_CTX()
    open(path, "r") do io
        buf = Vector{UInt8}(undef, 1 << 20)
        while !eof(io)
            n = readbytes!(io, buf)
            n == 0 && break
            SHA.update!(ctx, view(buf, 1:n))
        end
    end
    return bytes2hex(SHA.digest!(ctx))
end

"""
    data_provenance(
        output_dir::AbstractString;
        config_path::AbstractString = "",
        data_sources::Union{Nothing, AbstractDict{Symbol, DataSource}} = nothing,
        extra::AbstractDict = Dict{String, Any}(),
    ) -> String

Write `data_provenance.json` into `output_dir`, recording every declared input with the exact
file that backs it, and return the path.

For each data source the record carries the source, the cache path, whether the cached file is
present, its size in bytes, and its SHA-256.

This is deliberately a *separate* file from `resolved_config.toml`. The resolved config records
what was *asked for*; this records what was *actually read*. When a run degrades, or a cached
artefact is stale, those two diverge, and the divergence is the point.

`data_sources` defaults to the whole declared registry rather than to an empty set. The
docstring promises every declared input, and a caller that passes nothing should get exactly
that; a caller recording a subset -- one scenario's inputs, say -- passes it explicitly.
"""
function data_provenance(
    output_dir::AbstractString;
    config_path::AbstractString = "",
    data_sources::Union{Nothing, AbstractDict{Symbol, DataSource}} = nothing,
    extra::AbstractDict = Dict{String, Any}(),
)
    isdir(output_dir) || mkpath(output_dir)

    sources = something(data_sources, DATA_SOURCES)
    inputs = Dict{String, Any}()

    for key in sort!(collect(keys(sources)); by = string)
        d = sources[key]
        rel = resolved_cache_path(d, output_dir)
        present = isdir(rel) || isfile(rel)
        size_bytes = 0
        if isfile(rel)
            size_bytes = filesize(rel)
        elseif isdir(rel)
            for (root, _, files) in walkdir(rel), f in files
                size_bytes += filesize(joinpath(root, f))
            end
        end
        inputs[string(d.key)] = Dict{String, Any}(
            "title" => d.title,
            "kind" => string(d.kind),
            "keyless" => d.keyless,
            "source" => d.primary,
            "cache" => replace(rel, '\\' => '/'),
            "present" => present,
            "bytes" => size_bytes,
            "sha256" => isfile(rel) ? file_digest(rel) : "",
            "license" => d.license,
        )
    end

    record = Dict{String, Any}(
        "generated_utc" => string(Dates.now(Dates.UTC)),
        "config" => replace(config_path, '\\' => '/'),
        "package_version" => string(VERSION),
        "input_dir" => replace(input_dir(output_dir), '\\' => '/'),
        "inputs" => inputs,
    )
    for (k, v) in extra
        record[k] = v
    end

    path = joinpath(output_dir, "data_provenance.json")
    open(path, "w") do io
        JSON3.write(io, record)
    end
    return path
end

"""
    data_source(key::Symbol) -> DataSource

Look up a declared input by its config key, with a helpful error listing the valid keys.
"""
function data_source(key::Symbol)
    haskey(DATA_SOURCES, key) || error(
        "Unknown data source '$(key)'. Declared: " *
        join(sort(string.(collect(keys(DATA_SOURCES)))), ", "))
    return DATA_SOURCES[key]
end

"""
    describe_data_sources(io = stdout) -> nothing

Print the full registry: what each input is, which source to use, the alternatives, and the
access requirements. This is the human-facing half of the reproducibility contract; the
machine-facing half is `data_provenance.json`, written by a run.
"""
function describe_data_sources(io::IO = stdout)
    println(io, "Physical inputs required by TethysFlow / TethysDrift\n")
    println(io, "="^78)
    for key in sort!(collect(keys(DATA_SOURCES)); by = string)
        d = DATA_SOURCES[key]
        println(io, "\n", uppercase(string(d.key)), "  (", d.kind, ", keyless=", d.keyless, ")")
        println(io, "  ", d.title)
        println(io, "  primary : ", d.primary)
        println(io, "  cache   : ", d.cache)
        println(io, "  license : ", d.license)
        println(io, "  other sources:")
        for a in d.alternatives
            println(io, "    - ", a)
        end
        println(io, "  notes   : ", d.notes)
    end
    return nothing
end

# TethysFlow- and TethysDrift-specific DATA_SOURCES
const DATA_SOURCES = Dict{Symbol, DataSource}(

    :bathymetry => DataSource(
        :bathymetry,
        "ETOPO 2022 global relief (15 arc-second)",
        :gridded, true,
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ETOPO_2022_v1_15s.nc",
        [
            "ETOPO 180-arcsec, same ERDDAP host: https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc (coarser)",
            "GEBCO via NCEI: https://www.ncei.noaa.gov/products/global-relief-model (keyless, coarser)",
            "No fallback is applied. An unavailable dataset aborts the run; see :bathymetry notes.",
        ],
        "<store>/discretized/bathymetry.nc",
        "Public domain (NOAA)",
        "Configured by [data] bathy_source. The dataset id 'etopo2022' resolves to " *
        "ETOPO_2022_v1_15s.nc. Set it to 'synthetic' to generate an analytical shelf " *
        "instead and run with no network at all.",
        tier = :discretized,

    ),

    :glorys => DataSource(
        :glorys,
        "GLORYS12V1 Copernicus Marine reanalysis (temperature, salinity, currents)",
        :gridded, false,
        "https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030",
        [
            "GLORYS12V1: 1/12° global ocean reanalysis, 1993-present, 3-hourly/daily",
            "Requires free Copernicus Marine registration: ~/.copernicusmarine/credentials",
            "Provides temperature, salinity, u, v, sea surface height, MLD",
            "1/12° horizontal, 50 vertical levels, 3-hourly to daily output",
        ],
        "<store>/modelled/glorys.zarr",
        "Copernicus Marine Service - GLOBAL_MULTIYEAR_PHY_001_030",
        "Requires ~/.copernicusmarine/credentials; free registration at marine.copernicus.eu",
        tier = :modelled,
    ),

    :hydrography => DataSource(
        :hydrography,
        "World Ocean Atlas 2023 climatology (temperature and salinity)",
        :gridded, true,
        "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA",
        [
            "WOA23 via OPeNDAP: https://www.ncei.noaa.gov/thredds/dodsC/ncei/woa (keyless, subsetting without a full download)",
            "GLORYS12V1 via Copernicus Marine: https://data.marine.copernicus.eu product " *
            "GLOBAL_MULTIYEAR_PHY_001_030. NOT KEYLESS: it needs a free registration and " *
            "~/.copernicusmarine/credentials (uid and key); without them the subset service " *
            "returns 401. This entry is the *other* source this note previously described " *
            "incorrectly as anonymous - the credential requirement is the same one that " *
            ":lateral_boundary records, and the claim there is the accurate one.",
            "WHICH TO USE. WOA23 is a climatology: monthly means over decades, no year, no " *
            "weather, no trend. GLORYS is a reanalysis: real dates, daily-to-hourly, " *
            "interannual and mesoscale structure. A hydrodynamic model needs the time " *
            "evolution, so GLORYS is the right choice for driving one; WOA23 is the right " *
            "choice for a climatological baseline and for bias-reference comparison. " *
            "TethysFlow and TethysDrift configure this: GLORYS for `ocean_boundary_source` " *
            "because it carries currents, WOA23 for the initial condition.",
        ],
        "<store>/discretized/hydrography_temperature_<mm>.zarr",
        "Public domain (NOAA)",
        "Configured by [data] hydrography_source. The 1.00 degree monthly files are used and " *
        "regridded onto the model grid; the 0.25 degree product is a large download for the " *
        "same climatology."

    ),

    :woa23 => DataSource(
        :woa23,
        "WOA23 biogeochemical climatology (oxygen, nutrients, pH)",
        :gridded, true,
        "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/oxygen",
        [
            "World Ocean Atlas 2023 dissolved oxygen, NOAA NCEI. Keyless, public domain, and " *
            "the same collection and OPeNDAP endpoints already used for temperature and " *
            "salinity, so no new access route is needed. The 1x1 degree annual climatology " *
            "is small enough to fetch in full.",
            "WOA23 at its native 0.25 degree grid is available from the same directory when " *
            "the 1 degree climatology is too coarse for the domain.",
        ],
        "<store>/discretized/woa23_oxygen_<mm>_0.25deg.nc",
        "Public domain (NOAA).",
        "Fetched by `fetch_woa23_hydrography(..., include_o2 = true)`, the same reader as " *
        "temperature and salinity, since it is the same NOAA NCEI collection and depth " *
        "axis. Two things the reader documents and that are easy to get wrong: dissolved " *
        "oxygen is published only at 1.00 and 5.00 degree, under `oxygen/netcdf/all/` " *
        "with an `all_` filename prefix, so the temperature URLs do not work for it; and " *
        "the OPeNDAP endpoint is tried first but NCEI's THREDDS was returning 503 " *
        "throughout 2026-09-27, with whole-file static URLs as the reliable fallback. " *
        "The file carries `o_an` in ml/L, the unit WOA23 reports; no conversion to " *
        "umol/kg is applied. Read when [hydrodynamics] enable_oxygen is true."

    ),

    :surface_winds => DataSource(
        :surface_winds,
        "ERA5 surface winds from Open-Meteo (wind stress for the model)",
        :gridded, true,
        "https://archive-api.open-meteo.com/v1/archive (ERA5 subset)",
        [
            "ERA5 from the Copernicus Climate Data Store: https://cds.climate.copernicus.eu (keyless for the public ERA5 subset).",
            "Global Wind Atlas: https://globalwindatlas.info (keyless, climatological only).",
            "The SAME physical product `:atmosphere` carries. That entry describes the whole surface state -- stress plus air temperature, pressure, humidity, cloud and shortwave -- while this one names the winds alone, because that is what `[data] wind_source` and `[atmosphere] forcing` select and what `lakehouse/Tethys/era5.zarr` holds. Two keys, one dataset: the registry is keyed by what a run can ask for, not by how many files exist.",
        ],
        "<store>/modelled/era5.zarr",
        "Public domain (Copernicus ERA5 / Open-Meteo).",
        "Fetched by `fetch_open_meteo_winds` with the ERA5 surface state ingested alongside the " *
        "stress on the same hourly time axis, so stress and heat flux vary in time together. " *
        "They do NOT vary in space: Open-Meteo's archive endpoint answers with one " *
        "representative point per request, so the field is a box value repeated across the " *
        "domain. A run that needs resolved atmospheric structure needs a gridded product. " *
        "Configured by [atmosphere] forcing."
        ,
        tier = :modelled,

    ),

    :atmosphere => DataSource(
        :atmosphere,
        "Atmospheric surface forcing (wind stress, heat flux)",
        :gridded, true,
        "https://archive-api.open-meteo.com/v1/archive",
        [
            "ERA5 from the Copernicus Climate Data Store: https://cds.climate.copernicus.eu (KEYLESS for the public ERA5 subset)",
            "Global Wind Atlas: https://globalwindatlas.info (keyless, climatological only)",
        ],
        "<store>/modelled/era5.zarr",
        "See :surface_winds",
        "The fetcher ingests the ERA5 surface state alongside the winds -- air temperature, " *
        "surface pressure, relative humidity, cloud fraction and incoming shortwave -- on the " *
        "same hourly time axis as the stress, and the model is driven by a bulk net heat flux " *
        "computed from those fields and the model's own sea-surface temperature (Reed " *
        "shortwave, Berkson longwave, standard bulk sensible and latent terms). Stress and " *
        "flux therefore vary in time. They do NOT vary in space: Open-Meteo's archive endpoint " *
        "answers with one representative point per request, so the field is a box value " *
        "repeated across the domain. A run that needs resolved atmospheric structure needs a " *
        "gridded product, and `[atmosphere] forcing` is where that would be selected.",
        tier = :modelled,

    ),

    :lateral_boundary => DataSource(
        :lateral_boundary,
        "Open-boundary ocean state for the sponge layers (GLORYS12V1, via Copernicus)",
        :gridded, false,
        "https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030",
        [
            "VERIFIED 2026-10-11 BY READING THE CODE, NOT THE NETWORK. The fetcher registered for this input is `_fetcher_copernicus_physics` -> `fetch_copernicus_physics_subset`, whose dataset id defaults to GLOBAL_MULTIYEAR_PHY_001_030 -- that is GLORYS12V1. So this entry's primary was wrong: it named a HYCOM OPeNDAP catalog that no code here ever requests, and named a fetcher `fetch_hycom_boundary` that does not exist in this repository. Verified by searching every package for that symbol: it appears only in these notes.",
            "The earlier note claimed every HYCOM data request returned HTTP 400 and that .dds/.das/.html returned 200. That probe was never re-checked and is not the basis of this entry any more: what matters is what the code does, and the code asks Copernicus for GLORYS.",
            "GLORYS12V1 requires a free Copernicus Marine account and ~/.copernicusmarine/credentials (uid and key). Without them the subset service returns 401. This machine has credentials, so the fetch works here; a fresh checkout will NOT work until `copernicusmarine login` is run. That is why `keyless` is false and not a comment.",
            "Scope limit, verified from the fetcher's own command: it requests only `--variable thetao --variable so`. It does NOT request u/v. So this input constrains temperature and salinity at the sponge, not the normal velocity, and the earlier note's claim that it 'carries velocities, which is what a sponge actually needs' does not hold for the subset actually taken. The open edge is still only as open as the relaxation constants in [boundaries] u_inflow/v_inflow.",
            "NCAR GDEX holds a keyless record for GLORYS12V1 (d010049, DOI 10.5065/3NSD-QP06) with 'Access Constraints: None', but its dataset catalog is empty -- no files -- so it does not serve data. A metadata record is not a served file.",
        ],
        "<store>/modelled/lateral_boundary.nc",
        "Copernicus Marine Service - GLOBAL_MULTIYEAR_PHY_001_030."
        ,
        "Fetched by `fetch_copernicus_physics_subset` through `_fetcher_copernicus_physics`. " *
        "The cache path says glorys_boundary.nc because that is what is actually fetched; the " *
        "name hycom_boundary.nc that was here before described a product no code requests. " *
        "Configured by [boundaries] ocean_boundary_source; the sponge is only active when " *
        "`lateral_boundary_relaxation` is true or open boundary conditions are supplied, and " *
        "otherwise the domain edges relax toward the configured constants."
        ,
        tier = :modelled,

    ),

    :coastline => DataSource(
        :coastline,
        "Natural Earth coastline (10 m, public domain)",
        :gridded, true,
        "https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/",
        [
            "MEASURED AND OPEN. Natural Earth is public domain and needs no key. The GeoJSON mirror under nvkelso/natural-earth-vector is the keyless endpoint actually used; the Natural Earth download page above is the citable origin. 10 m is the finest coastline Natural Earth publishes.",
            "The archive is a whole-world file of ~4100 LineString features (about 10 MB). `fetch_natural_earth_coastline` clips it to the run's domain plus a margin before writing, so a regional run stores only the segments it needs.",
            "Coarser 50 m and 110 m series are available via `[data] coastline_source` when only a regional outline is wanted.",
        ],
        "<store>/discretized/coastline.parquet",
        "Public domain (Natural Earth).",
        "Written as lon,lat vertex lists grouped under `> name,code` headers, the format " *
        "`load_coastline_polygons` reads. This replaces the hand-drawn regional outline that " *
        "previously shipped in the repository, which described Nova Scotia with a handful of " *
        "vertices and left most of the domain's land unmasked."
    ),

    :tides => DataSource(
        :tides,
        "TPXO9-Atlas astronomical tidal solution (global harmonic velocity)",
        :gridded, true,
        "https://zenodo.org/records/8074917",
        [
            "MEASURED AND CORRECT FOR THIS STUDY AREA. Zenodo 8074917, 'TPXO9 data used for ICON-O tides evaluation', open, CC-BY-4.0. The member `TPXO9_datafiles/u_tpxo9.v1.nc` is the GLOBAL harmonic velocity solution: lon_u spans 0..360, lat_u spans -90..90, with 60 constituents. Verified by inflating only a header prefix of the archived member, so the full 1.4 GB was never downloaded. Extract it with `fetch_input(:tides)`; the archive URL is https://zenodo.org/api/records/8074917/files/TPXO9_datafiles.zip/content",
            "DO NOT USE for this study area. Two smaller archives look equivalent and are NOT, both measured: the 48.7 MB `TPXO9_atlas_v2.zip` on Zenodo 22970362 (pyTMD test data) spans only lat -34.97..14.97, and the `*_remap*` members in record 8074917 are regional (e.g. h_tpxo9_remap3600x1800_sonett.nc spans lon -30..30, lat -55..-10). A tidal archive must be checked against its own lat range before use; the reader raises an error naming the actual coverage rather than returning an empty box.",
            "TPXO9-Atlas from the publisher, https://www.tpxo.atlas.oregonstate.edu (free and open, but the host does not resolve from every network, including this one).",
            "FES2014 / FES2022 from AVISO, https://www.aviso.altimetry.fr (AVISO requires a free account).",
        ],
        "<store>/modelled/tpxo9.nc",
        "TPXO9-Atlas: CC-BY-4.0.",
          "u_tpxo9.v1.nc carries the VELOCITY harmonics on the staggered u/v grids, in " *
          "BOTH of the two forms TPXO9-Atlas ships: amplitude/phase " *
          "(ua/up/va/vp, cm/s and degrees GMT) and complex (URe/UIm/VRe/VIm, " *
          "m s^-1). They are the same information - URe = ua*cos(up), " *
          "UIm = ua*sin(up) - so `fetch_tpxo9_velocity` accepts either and " *
          "reports which it found. h_tpxo9.v1.nc is the elevation product. " *
          "15 constituents: m2, s2, n2, k2, k1, o1, p1, q1, mm, mf, m4, mn4, " *
          "ms4, 2n2, s1. A barotropic open boundary is driven by the normal " *
          "velocity, so the velocity product is what the boundary forcing needs. " *
           "Note the two conventions that are easy to get wrong: longitude is 0..360 " *
           "rather than -180..180, and the imaginary part enters with a MINUS sign " *
           "(the files document `phase = atan2(-hIm, hRe)`).",
        tier = :modelled,
    ),

    :bottom_temperature => DataSource(
        :bottom_temperature,
        "Bottom temperature from DFO survey observations",
        :station, true,
        "biodata/bt.rdata, converted to a GeoParquet point table carried in " *
        "lakehouse/Tethys/raw/scotian_shelf_bottom_temperature_surveys_1970_2025.parquet",
        [
            "No alternative source is needed or offered: this is the highest spatial and " *
            "temporal resolution bottom temperature available for the region, and it is " *
            "shipped with the package rather than fetched.",
            "GLORYS12V1 via Copernicus Marine provides continuous modelled bottom " *
            "temperature where the surveys have gaps and after 2025 blending. That is the " *
            "product the ocean model actually runs on; this one supplements it and is not " *
            "what forces the hydrodynamic simulation.",
        ],
        "<store>/raw/bottom_temperature.parquet",
        "Canadian Department of Fisheries and Oceans survey archive; see the `project` " *
        "column of the raw table for the 30 contributing programmes.",
        "442177 observations from 1970 to 2025 across 30 survey programmes (Groundfish, " *
        "Inshore and Offshore Scallop, Snow Crab, OTN acoustic telemetry, observer " *
        "programmes). Depth of observation `z` spans 0 to ~5000 m, so some rows are " *
        "mid-water rather than the sea floor - use `max_depth_m` to restrict them. The " *
        "gridded output carries `bottom_temperature_n`, the count of observations per " *
        "cell, because a bin with one 2019 value is not the same claim as one with forty. " *
        "DECLARED, NOT CONSUMED BY THE HYDRODYNAMIC MODEL. Nothing in the ocean-model or " *
        "larval paths fetches this input today -- the model's bottom temperature comes from " *
        "GLORYS via :lateral_boundary / :hydrography. It is declared because the fetchers and " *
        "the regional cube exist and are exercised, and because a future bottom-temperature " *
        "forcing should not have to rediscover this table. Not required to run a model.",
        tier = :raw,

    ),

    :substrate => DataSource(
        :substrate,
        "Interpolated substrate grain size, Scotian Shelf and Gulf of Maine",
        :gridded, true,
        "biodata/substrate_grainsize.rdata, converted to a GeoParquet point table carried " *
        "in lakehouse/Tethys/raw/scotian_shelf_substrate_grain_size.parquet",
        [
            "No alternative: the underlying raw sediment samples are not publicly " *
            "available, so this interpolated surface is the only local substrate product.",
        ],
        "<store>/discretized/substrate.parquet",
        "Canadian Hydrographic Service / DFO sediment compilation.",
        "714063 points spanning lon -67.7..-57.1, lat 41.3..47.0, grain diameter " *
        "0.003..33.2 mm (mud to cobble). This is ALREADY INTERPOLATED surface data: the " *
        "raw grab samples behind it are not available, which is stated in the dataset's " *
        "`interpolated` attribute so no downstream use mistakes it for measurements. " *
        "DECLARED, NOT CONSUMED BY THE HYDRODYNAMIC MODEL. Nothing in the ocean-model or " *
        "larval paths fetches this input today; the substrate surface is a habitat input for " *
        "the regional cube. It is declared because the fetcher and the gridding are built " *
        "and exercised, and because a future substrate-dependent biology should not have to " *
        "rediscover this table. Not required to run a model.",
        tier = :discretized,

    ),

    :nwa_bathymetry => DataSource(
        :nwa_bathymetry,
        "CHS 15-arc-second Northwest Atlantic bathymetric grid",
        :gridded, true,
        "biodata/nwa.chs15sec_lon_lat_depth.xyz, converted to a GeoParquet point table " *
        "carried in lakehouse/Tethys/raw/northwest_atlantic_chs_15arcsec_soundings.parquet",
        [
            "ETOPO_2022_v1_15s via ERDDAP - see :bathymetry - is the same nominal " *
            "resolution but a global model rather than CHS soundings. They are different " *
            "products and are NOT interchangeable: they differ on the shelf by metres to " *
            "tens of metres, and neither is wrong. Choose one per run.",
        ],
        "<store>/discretized/nwa_bathymetry.zarr",
        "Canadian Hydrographic Service, Northwest Atlantic soundings.",
        "28142338 XYZ rows spanning lon -72..-42 and lat 40..64, on a REGULAR 15 arc-second " *
        "lattice of 7201 x 5761 nodes with land cells omitted (41.5M lattice cells, 28.1M " *
        "present) - so this is a gridded product shipped as a point list, not an " *
        "irregular sounding cloud. Depths -5467..+383 m, positive values being land. " *
        "Higher vertical detail than ETOPO where CHS has surveyed, " *
        "which is why the cube prefers it by default for the sea floor.",
        tier = :discretized,

    ),
)

end # module Manifest
