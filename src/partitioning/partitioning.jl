"""
    partitioning.jl

Spatial and spatiotemporal domain decomposition, Voronoi/CVT/KVT/QVT/BVT/HVT/AVT tessellations,
hexagonal meshes, spatial adjacency graphs, BYM2 scaling, and cross-validation blocks.
"""
module Partitioning

using
    Clustering,
    CoordRefSystems,
    DataFrames,
    Dates,
    DelaunayTriangulation,
    Graphs,
    JLD2,
    LibGEOS,
    LinearAlgebra,
    NearestNeighbors,
    Shapefile,
    SparseArrays,
    StaticArrays,
    Statistics,
    StatsBase,
    Unitful

const _HAS_COORDSYS = true

include("columns.jl")
include("geometry.jl")
include("tessellation.jl")
include("spatial_units.jl")
include("spatial_graphs.jl")
include("temporal.jl")
include("hexmesh.jl")
include("hsi.jl")

export
    STANDARD_SPATIAL_COORDINATE_PAIRS,
    STANDARD_TEMPORAL_CANDIDATES,
    assign_spatial_units,
    assign_spatial_units_inferred,
    assign_time_units,
    assign_spatiotemporal_units,
    discretize_data,
    spatial_block_cv,
    expand_hull,
    build_hex_mesh,
    map_point_to_units,
    lonlat_to_xy_km,
    xy_km_to_lonlat,
    spatial_weights_matrix,
    spatial_knn_graph,
    spatial_radius_graph,
    scaling_factor_bym2,
    compute_network_transfer_matrix,
    build_monthly_hsi_matrix,
    load_hsi_jld2,
    reshard_hsi_field,
    poly_xy,
    centroids_xy,
    extract_polygons,
    extract_centroids,
    build_spatial_domain_grid,
    reconstruct_spatial_surface,
    extract_graph_edges_and_degrees

end # module Partitioning
