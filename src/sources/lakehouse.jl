"""
    lakehouse.jl

Lakehouse storage organization, tier management, and high-level ingestion
and publishing workflows for Tethys.jl.

Standardizes storage directory hierarchy across:
- Bronze (:raw): Immutable upstream source files, raw netCDF/CSV downloads.
- Silver (:processed): Quality-controlled, cleaned, regularized Zarr/GeoParquet grids.
- Gold (:derived): Consumer-produced model outputs, dispersal kernels, GAM predictions.
"""
module LakehouseModule

using ..Types
using ..Types: GeoDataset, GeoArray, CoordinateSystem, Dimension
using ..CatalogModule: GeoDataCatalog, DatasetEntry, default_catalog_path,
    register_dataset!, save_catalog, geofetch
# The lakehouse root is declared by the catalogue module, and one tier's layout must not own
# the root: a submodule cannot see a sibling's name unqualified, so calling `lakehouse_root_dir`
# here without importing it raised UndefVarError on the first `lakehouse_path`. `lakehouse_path`
# is pure -- it never mkpath -- which is exactly why the root has to be resolvable without
# touching the filesystem.
using ..CatalogModule: lakehouse_root_dir
using Dates
using UUIDs
using SHA

export lakehouse_path,
       lakehouse_tiers,
       geopublish_dataset!,
       geofetch_dataset,
       compute_lakehouse_checksum,
       LakehouseTier

"""
    LakehouseTier

Enumeration of supported medallion architectural tiers.
"""
@enum LakehouseTier begin
    TIER_RAW        # Bronze: raw downloads / observations
    TIER_PROCESSED  # Silver: harmonized, QC'd, standard grids
    TIER_DERIVED    # Gold: model predictions, dispersal kernels, predictions
end

"""
    tier_symbol(t::LakehouseTier) -> Symbol
"""
function tier_symbol(t::LakehouseTier)::Symbol
    t == TIER_RAW && return :raw
    t == TIER_PROCESSED && return :processed
    t == TIER_DERIVED && return :derived
    return :unknown
end

"""
    lakehouse_tiers() -> Vector{Symbol}

Return list of supported lakehouse tier symbols.
"""
lakehouse_tiers() = [:raw, :processed, :derived]

"""
    lakehouse_tier_symbol(t::LakehouseTier) -> Symbol
"""
lakehouse_tier_symbol(t::LakehouseTier) = tier_symbol(t)

"""
    lakehouse_path(key::Symbol; tier::Symbol = :processed, ext::String = "zarr") -> String

Return the standard filesystem destination path for a given dataset in the lakehouse.

Path layout: `<lakehouse_root_dir>/<tier>/<key>.<ext>`

This function is pure: it does not create any directories on disk.
"""
function lakehouse_path(key::Symbol; tier::Symbol = :processed, ext::String = "zarr")
    tier in lakehouse_tiers() || throw(ArgumentError(
        "Invalid lakehouse tier :$tier. Supported tiers: $(lakehouse_tiers())"
    ))
    root = lakehouse_root_dir()
    tier_dir = joinpath(root, string(tier))
    clean_ext = startswith(ext, ".") ? ext[2:end] : ext
    return joinpath(tier_dir, "$(key).$(clean_ext)")
end
"""
    compute_lakehouse_checksum(path::String) -> String

Compute SHA-256 digest of a local file or directory (e.g. Zarr store).

The digest includes a schema header (v1) so that future schema changes
produce a different digest even for identical file content.
"""
function compute_lakehouse_checksum(path::String)::String
    if !ispath(path)
        return ""
    end
    ctx = SHA.SHA2_256_CTX()
    # Schema header v1
    SHA.update!(ctx, Vector{UInt8}("LAKEHOUSE_CHECKSUM_V1\n"))
    if isfile(path)
        SHA.update!(ctx, Vector{UInt8}("FILE\n"))
        open(path, "r") do io
            while !eof(io)
                SHA.update!(ctx, read(io, 65536))
            end
        end
    elseif isdir(path)
        # Aggregate hash over sorted constituent files
        for (root, _, files) in walkdir(path)
            for file in sort(files)
                fpath = joinpath(root, file)
                rel = relpath(fpath, path)
                SHA.update!(ctx, Vector{UInt8}(rel))
                open(fpath, "r") do io
                    while !eof(io)
                        SHA.update!(ctx, read(io, 65536))
                    end
                end
            end
        end
    end
    # `digest!` is what finalises the context and returns the bytes. The field named `digest`
    # does not exist on a SHA2 context (it holds `state`, `bytecount`, `buffer`, `used`), so
    # this raised a FieldError for every file it was asked to hash -- and nothing called it,
    # because a checksum nobody asks for is a checksum nobody tests.
    return bytes2hex(SHA.digest!(ctx))
end

"""
    geopublish_dataset!(catalog::GeoDataCatalog,
                        location::String;
                        key::Symbol,
                        name::String,
                        tier::Symbol = :processed,
                        producer::String = "",
                        derived_from::Vector{Symbol} = Symbol[],
                        format::Symbol = :zarr,
                        variables::Vector{String} = String[],
                        spatial_bounds::Tuple{Float64, Float64, Float64, Float64} = (0.0, 0.0, 0.0, 0.0),
                        temporal_coverage::Tuple{DateTime, DateTime} = (DateTime(1970,1,1), DateTime(1970,1,1)),
                        notes::String = "",
                        compute_hash::Bool = false,
                        save_now::Bool = true) -> DatasetEntry

Publish and register a dataset with lakehouse tier and lineage metadata.
"""
function geopublish_dataset!(catalog::GeoDataCatalog,
                             location::String;
                             key::Symbol,
                             name::String,
                             tier::Symbol = :processed,
                             producer::String = "",
                             derived_from::Vector{Symbol} = Symbol[],
                             format::Symbol = :zarr,
                             variables::Vector{String} = String[],
                             spatial_bounds::Tuple{Float64, Float64, Float64, Float64} = (0.0, 0.0, 0.0, 0.0),
                             temporal_coverage::Tuple{DateTime, DateTime} = (DateTime(1970,1,1), DateTime(1970,1,1)),
                             notes::String = "",
                             compute_hash::Bool = false,
                             save_now::Bool = true)::DatasetEntry

    checksum = compute_hash && ispath(location) ? compute_lakehouse_checksum(location) : ""

    entry = DatasetEntry(
        key = key,
        name = name,
        tier = tier,
        producer = producer,
        derived_from = derived_from,
        format = format,
        location = location,
        variables = variables,
        spatial_bounds = spatial_bounds,
        temporal_coverage = temporal_coverage,
        checksum = checksum,
        notes = notes
    )

    register_dataset!(catalog, entry)
    if save_now && !catalog.auto_save && !isempty(catalog.catalog_file)
        save_catalog(catalog)
    end

    return entry
end

"""
    geofetch_dataset(catalog::GeoDataCatalog, key::Symbol) -> DatasetEntry

Retrieve a registered lakehouse dataset entry by key and bump its access count.
"""
function geofetch_dataset(catalog::GeoDataCatalog, key::Symbol)::DatasetEntry
    return geofetch(catalog, key)
end

end # module LakehouseModule
