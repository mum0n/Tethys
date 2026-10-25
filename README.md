# Tethys

Foundational geospatial data model, coordinate reference systems, hierarchical analytical lakehouse storage, automated spatial and spatiotemporal partitioning, and environmental data ingestion engine for the Tethys ecosystem.

For the complete API specification, mathematical foundations, architectural review, and reproducible examples, see:

- [**Tethys Technical Specification & Method Reference**](Tethys.md)

---

## Subsystem Highlights

- **Core Multidimensional Data Model**: Backend-agnostic `GeoDataset` and `GeoArray` with formal axis `Dimension` metadata and `CoordinateSystem`.
- **Hierarchical Lakehouse Storage**: Medallion architecture (bronze, silver, gold) with pluggable backends for Zarr, NetCDF/NCZarr (`NCDatasets.jl`), and GeoParquet (`Parquet2.jl`), managed via `GeoStorage`.
- **Geodesy & Projections**: Native integration with `CoordRefSystems.jl` supporting Geographic, Universal Transverse Mercator (UTM), Lambert Conformal Conic, Mercator, and shelf-aligned rotated oceanographic grid frames.
- **Computational Geometry & Marine Masking**: High-resolution shoreline buffering, polygon containment testing via ray-casting, bathymetric smoothing to constrain barotropic gravity wave speed, and hypertidal dampening.
- **Physical Data Ingestion**: Automated fetchers with SHA-256 cryptographic provenance verification for ETOPO 2022 bathymetry, CHS soundings, Copernicus Marine GLORYS12V1 physics, WOA23 hydrography, Open-Meteo surface winds, and TPXO9 tidal velocity atlas.
- **Spatial & Spatiotemporal Partitioning**: Hexagonal honeycomb tessellation, Centroidal Voronoi Tessellation (CVT) via Lloyd's relaxation, K-means partitioning, connected spatial graph construction (`spatial_weights_matrix`), island bridging via KD-tree nearest neighbors, and BYM2 spectral scaling factor ($\tau$) computation.
- **Visualization Layer (`TethysPlots.jl`)**: Dedicated plotting and mapping library providing publication-ready static vector figures (`Plots.jl`) and interactive Leaflet GIS dashboards (`Leaflet.js`).
- **Catalog & Provenance**: `DatasetCatalog` registry and deterministic pipeline artifact management (`purge_run_state`).

---

## Installation

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

---

## Basic Usage

```julia
using Tethys

# Build hexagonal mesh over study area
mesh = build_hex_mesh((-65.0, -60.0, 42.0, 45.0); cell_size_km = 30.0)

# Build sparse spatial weights matrix
W = spatial_weights_matrix(mesh.centroids_lonlat; method = :knn, k = 6)

# Compute BYM2 scaling factor for spatial priors
tau = scaling_factor_bym2(W)
```
