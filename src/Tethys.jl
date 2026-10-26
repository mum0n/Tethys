"""
    Tethys.jl

Consolidated geospatial foundation layer unifying:
- Backend-agnostic n-dimensional array containers and dataset algebra
- Standardized configuration management and command-line execution dispatch
- Geodetic projections, transformations, and vector rotations via `CoordRefSystems.jl`
- Cloud-native storage drivers (Zarr, NetCDF/NCZarr, GeoParquet) and lakehouse registries
- Data acquisition drivers (ETOPO, GEBCO, NOAA ERDDAP, WOA23, TPXO9, Copernicus, Open-Meteo)
- Spatial and spatiotemporal domain decomposition, tessellations, and graph topologies

# Design Principles
- **Streamlined Architecture**: Cohesive single-package deployment removing inter-package
  version pinning locks and redundant metadata translations.
- **Consistent Verb-Object Semantics**: Standard operations (`load_dataset`, `save_dataset`,
  `slice_dataset`, `sample_dataset`, `regrid_dataset`, `build_projection`, `load_config`,
  `parse_cli_args`, `assign_spatial_units`).
- **Explicit Failure & Strict Checking**: Missing coordinate metadata or unsupported schemes
  raise informative exceptions rather than silent fallbacks or invented coordinates.
"""
module Tethys

using Reexport
using Base: @kwdef

# Standard libraries
using Base64
using CodecZlib
using Dates
using LinearAlgebra
using Random
using Serialization
using SHA
using SparseArrays
using Statistics
using TOML
using TranscodingStreams
using UUIDs

# External geospatial and scientific dependencies
using ArchGDAL
using CFTime
using Clustering
using CoordRefSystems
using DataFrames
using Downloads
using GeoFormatTypes
using GeoInterface
using GeoJSON
using GeoParquet
using HTTP
using Interpolations
import Interpolations: interpolate, extrapolate, Gridded, Linear, Flat
using JLD2
using JSON3
using LibGEOS
using NCDatasets
using NearestNeighbors
using Shapefile
using StaticArrays
using StatsBase
using Unitful
using WellKnownGeometry
using Zarr
using ZipFile

# ==============================================================================
# Core Subsystems
# ==============================================================================

include("core/types.jl")
@reexport using .Types

include("core/backends.jl")
@reexport using .Backends

include("core/config.jl")
@reexport using .Config

include("core/cli.jl")
@reexport using .CLI

include("core/coordinates.jl")
@reexport using .Coordinates

include("core/crs.jl")
@reexport using .CRS

include("core/operations.jl")
@reexport using .Operations

# Computational geometry and land/bathymetry classification
include("core/opengrid.jl")
include("core/landmask.jl")
include("core/bathy.jl")

# ==============================================================================
# Storage Drivers & Backend Registry
# ==============================================================================

include("storage/StorageRegistry.jl")
@reexport using .StorageRegistry

include("storage/common.jl")
include("storage/zarr_backend.jl")
include("storage/ncdatasets_backend.jl")
include("storage/geoparquet_backend.jl")

# Backend capability fallback
backend_capabilities(::GeoBackend) = BackendCapabilities(read = true, write = true)

# ==============================================================================
# High-Level Functional API
# ==============================================================================

include("api/utils.jl")
include("api/load.jl")
include("api/save.jl")
include("api/slice.jl")
include("api/select.jl")
include("api/values.jl")
include("api/join.jl")
include("api/index.jl")
include("api/regrid.jl")
include("api/aggregate.jl")
include("api/storage.jl")
include("api/run_state.jl")

# ==============================================================================
# Acquisition Drivers, Provenance, and Catalogues
# ==============================================================================

include("sources/Access/Axis.jl")
include("sources/Fetch/Download.jl")
include("sources/storage.jl")
include("sources/coastline.jl")
include("sources/bathymetry.jl")
include("sources/biodata.jl")
include("sources/bundled_coastline.jl")
include("sources/winds.jl")
include("sources/woa23.jl")
include("sources/boundary.jl")
include("sources/regrid_utils.jl")
include("sources/Fetch/tides.jl")
include("sources/Fetch/copernicus.jl")
@reexport using .Copernicus

include("sources/manifest.jl")
@reexport using .Manifest

include("sources/regional_cube.jl")

include("sources/catalog.jl")
@reexport using .CatalogModule

include("sources/lakehouse.jl")
@reexport using .LakehouseModule

# ==============================================================================
# Spatial and Spatiotemporal Partitioning
# ==============================================================================

include("partitioning/partitioning.jl")
@reexport using .Partitioning

# ==============================================================================
# Unified Standard Aliases
# ==============================================================================

# Dataset operations
const load_dataset = geoload
const save_dataset = geosave
const slice_dataset = geoslice
const sample_dataset = geovalues
const regrid_dataset = georegrid
const select_dataset = geoselect
const join_datasets = geojoin
const aggregate_dataset = geoaggregate
const index_dataset = geoindex
const geosummary = dataset_summary
const data_stats = dataset_stats
const variable_stats = dataset_stats
const create_dataset_template = create_zarr_template

# Data sources operations
const load_bathymetry = load_bathymetry_geodata
const save_bathymetry = save_bathymetry_geodata
const load_coastline = load_coastline_geodata
const load_winds = load_wind_stress_geodata
const build_bulk_surface_flux = build_bulk_surface_flux_geodata
const fetch_boundary_hydrography = fetch_boundary_hydrography_geodata
const build_boundary_tracer_interpolators = build_boundary_tracer_interpolators_geodata
const publish_dataset! = geopublish_dataset!
const fetch_dataset = geofetch_dataset

# Partitioning operations
const build_hex_mesh_planar = build_hex_mesh

# Configuration & CLI operations
const get_config_val = get_config_value
const display_cli_summary = print_cli_help

# Projection & Geometry operations
const build_projection = create_projection
const project_forward = forward_project
const project_inverse = inverse_project
const reproject_field = reproject_to_geographic

# Land mask aliases
const is_point_on_land_geodata = is_point_on_land
const is_marine_water_geodata = is_marine_water
# Backend & Source Registration
# ==============================================================================

function _register_default_backends()
    register_backend(:zarr, ZarrBackend())
    register_backend(:nczarr, NCDatasetsBackend(; format = "nczarr"))
    register_backend(:ncdatasets, NCDatasetsBackend())
    register_backend(:geoparquet, GeoParquetBackend())
end

function __init__()
    _register_default_backends()

    # Register acquisition driver plugins for geoload / geoget
    register_source_provider!(:erddap, (uri; cache_dir = nothing, bounds = nothing, kwargs...) -> begin
        lon_range = bounds !== nothing ? bounds[1] : (-71.0, -53.0)
        lat_range = bounds !== nothing ? bounds[2] : (40.0, 48.5)
        out_path = cache_dir !== nothing ?
            joinpath(cache_dir, "bathymetry.zarr") :
            joinpath("inputs", "bathymetry.zarr")
        fetch_erddap_bathymetry(;
            lon_range = lon_range,
            lat_range = lat_range,
            output_path = out_path,
            kwargs...
        )
    end)

    register_source_provider!(:copernicus, (uri; cache_dir = nothing, bounds = nothing, kwargs...) -> begin
        fetch_copernicus_physics_subset(; kwargs...)
    end)

    register_source_provider!(:open_meteo, (uri; cache_dir = nothing, bounds = nothing, kwargs...) -> begin
        lon_range = bounds !== nothing ? bounds[1] : (-71.0, -53.0)
        lat_range = bounds !== nothing ? bounds[2] : (40.0, 48.5)
        out_path = cache_dir !== nothing ?
            joinpath(cache_dir, "surface_winds.zarr") :
            joinpath("inputs", "surface_winds.zarr")
        fetch_open_meteo_winds(;
            lon_range = lon_range,
            lat_range = lat_range,
            output_path = out_path,
            kwargs...
        )
    end)

    register_source_provider!(:woa23, (uri; cache_dir = nothing, bounds = nothing, kwargs...) -> begin
        lon_range = bounds !== nothing ? bounds[1] : (-71.0, -53.0)
        lat_range = bounds !== nothing ? bounds[2] : (40.0, 48.5)
        out_dir = cache_dir !== nothing ? cache_dir : store_path("woa23")
        fetch_woa23(;
            lon_range = lon_range,
            lat_range = lat_range,
            output_dir = out_dir,
            kwargs...
        )
    end)

    register_source_provider!(:tides, (uri; cache_dir = nothing, bounds = nothing, kwargs...) -> begin
        out_dir = cache_dir !== nothing ? cache_dir : "inputs"
        fetch_input(:tides, out_dir; kwargs...)
    end)

    register_source_provider!(:natural_earth, (uri; cache_dir = nothing, bounds = nothing, kwargs...) -> begin
        lon_range = bounds !== nothing ? bounds[1] : (-71.0, -53.0)
        lat_range = bounds !== nothing ? bounds[2] : (40.0, 48.5)
        out_path = cache_dir !== nothing ?
            joinpath(cache_dir, "coastline.dat") :
            joinpath("inputs", "coastline.dat")
        fetch_natural_earth_coastline(;
            lon_range = lon_range,
            lat_range = lat_range,
            output_path = out_path,
            kwargs...
        )
    end)
end

# Re-export public API symbols
export
    # Core types
    DimensionType, DIM_SPATIAL, DIM_TEMPORAL, DIM_PARAMETRIC, DIM_GENERIC,
    Dimension, CoordinateSystem, GeoArray, GeoDataset, GeoBackend, BackendCapabilities,
    is_cartesian,

    # Configuration & CLI
    load_config, save_config, get_config_val, get_config_value, resolve_config_path, validate_config,
    parse_cli_args, display_cli_summary, print_cli_help,

    # Coordinate Reference Systems & Projections
    AbstractProjection, GeographicProjection, CoordRefSystemsGeographicProjection,
    CoordRefSystemsProjection, UTMProjection,
    LambertProjection, MercatorProjection, RotatedFrame, ReprojectionGrid,
    is_planar, resolve_crs, create_projection, build_projection, forward_project, inverse_project,
    project_forward, project_inverse, grid_convergence_angle, geographic_to_rotated,
    rotated_to_geographic, rotate_vector, unrotate_vector, rotate_vector_to_grid, rotate_vector_to_geo,
    reproject_to_geographic, reproject_field, optimal_shelf_rotation_angle,
    utm_to_lonlat, lonlat_to_utm, extract_wkt, is_wgs84_wkt,
    CoordinateTransformer, build_coordinate_transformer, transform_point, transform_polygon,
    projection_from_crs, transform_coordinates, transform_vector_dataset, projection_signature,

    # Coordinates & operations
    axis, require_coord, to_positive_down, dim_permutation, normalize_depth,
    standardize_dimension_name, is_regular_grid, grid_spacing, find_coord_indices, slice_indices,
    bounding_box, variables_like, is_cached,

    # Computational geometry
    regrid_2d_field, regrid_3d_field, buffer_distance_to_degrees, expand_domain_with_buffer,
    point_in_polygon, is_point_on_land, is_point_on_land_geodata,
    is_marine_water, is_marine_water_geodata, load_coastline_polygons,
    smooth_bathymetry, get_bathymetry_interpolator, extract_marine_cells, sample_marine_cells,

    # Primary dataset API
    load_dataset, geoload, geoget, save_dataset, geosave, slice_dataset, geoslice,
    sample_dataset, geovalues, geovalue, geoprofile, geotimeseries,
    regrid_dataset, georegrid, georegrid_regular, select_dataset, geoselect,
    join_datasets, geojoin, geomerged, aggregate_dataset, geoaggregate, index_dataset, geoindex,
    dataset_stats, data_stats, variable_stats, dataset_summary, geosummary,
    create_dataset_template, create_zarr_template,

    # Backend management
    register_backend, get_backend, list_backends, infer_backend, strip_scheme,
    register_source_provider!, unregister_source_provider!, list_source_providers,

    # Hierarchical analytical storage & Lakehouse
    GeoStorage, open_geostorage, close_geostorage, create_storage_group, has_storage_group,
    write_storage_variable!, read_storage_variable, list_storage_variables, list_storage_groups,
    lakehouse_dir, set_lakehouse_dir!, project_lakehouse_path,
    storage_config, store_root, store_path, storage_file_path,
    lakehouse_path, lakehouse_tiers,

    # Catalogs & provenance
    GeoDataCatalog, DatasetCatalog, DatasetEntry, load_catalog!, save_catalog, register_dataset!,
    unregister_dataset!, update_dataset!, find_datasets, get_dataset,
    publish_dataset!, geopublish_dataset!, fetch_dataset, geofetch_dataset,
    DATA_SOURCES, DataSource, data_source, describe_data_sources, fetch_input,
    data_provenance, available_data, data_summary, buildable_sources,

    # Oceanographic and atmospheric data products
    load_bathymetry, save_bathymetry, load_bathymetry_geodata, save_bathymetry_geodata,
    bathymetry, bathymetry_dataset, bathymetry_field, bathymetry_arrays,
    regrid_bathymetry_from_etopo, etopo_bathymetry_field, preferred_bathymetry_source,
    SCOTIAN_SHELF_COASTLINE, load_coastline, load_coastline_geodata,
    load_winds, load_wind_stress_geodata, build_bulk_surface_flux, build_bulk_surface_flux_geodata,
    wind_speed_to_kinematic_stress, wind_speed_from_stress,
    load_woa23_interpolators, fetch_woa23,
    fetch_boundary_hydrography, fetch_boundary_hydrography_geodata,
    build_boundary_tracer_interpolators, build_boundary_tracer_interpolators_geodata,
    RegionalCubeConfig, load_cube_config, standard_ocean_depths, assimilate_regional_cube,

    # Spatial partitioning and graph topology
    STANDARD_SPATIAL_COORDINATE_PAIRS, STANDARD_TEMPORAL_CANDIDATES,
    assign_spatial_units, assign_spatial_units_inferred, assign_time_units,
    assign_spatiotemporal_units, discretize_data, spatial_block_cv,
    expand_hull, build_hex_mesh, build_hex_mesh_planar, map_point_to_units,
    lonlat_to_xy_km, xy_km_to_lonlat,
    spatial_weights_matrix, spatial_knn_graph, spatial_radius_graph, scaling_factor_bym2,
    compute_network_transfer_matrix,
    build_monthly_hsi_matrix, load_hsi_jld2, reshard_hsi_field,
    poly_xy, centroids_xy, extract_polygons, extract_centroids,
    build_spatial_domain_grid, reconstruct_spatial_surface, extract_graph_edges_and_degrees,

    # Run state tracking and purging
    RunStateTarget, run_state_targets, confirm_run_state_purge, apply_run_state_purge, purge_run_state

end # module Tethys
