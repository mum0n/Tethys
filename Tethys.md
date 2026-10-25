# Tethys.jl API Specification, Architectural Framework, and Method Reference

`Tethys.jl` is the foundational oceanographic, spatial data, analytical storage, and domain discretization engine for the Tethys ecosystem. The package provides backend-agnostic multidimensional geospatial datasets, standardized coordinate reference systems, hierarchical lakehouse storage, automated spatial and spatiotemporal partitioning, physical data fetchers, cross-mesh topological transfer operators, and computational geometry tools.

---

## Architectural Framework and Subsystems

The library is organized into five operational subsystems:

```
Tethys
├── Core
│   ├── Types           # GeoDataset, GeoArray, Dimension, CoordinateSystem
│   ├── Operations      # Slicing, sampling, joining, aggregation, dimension permutation
│   ├── CRS             # CoordRefSystems integration, UTM, LCC, Mercator, rotated frames
│   ├── Landmask        # Ray-casting, polygon clipping, buffer zones, marine masking
│   ├── Bathymetry      # Topographic floors, bathymetric interpolators, wave speed constraints
│   ├── OpenGrid        # Staggered Arakawa C-grid geometric operators and convergence
│   └── Config & CLI    # TOML configuration validation, path resolution, CLI parsing
├── Storage
│   ├── StorageRegistry # Format registration, scheme parsing, backend lookup
│   ├── ZarrBackend     # Chunked, compressed N-dimensional array storage (Zarr.jl)
│   ├── NCDatasets      # NetCDF-4 and NCZarr storage interfaces (NCDatasets.jl)
│   └── GeoParquet      # Columnar partitioned tabular and track storage (Parquet2.jl)
├── API
│   ├── Unified Dataset # load_dataset, save_dataset, slice_dataset, sample_dataset, regrid_dataset
│   ├── GeoStorage      # Hierarchical group and variable lakehouse read/write engine
│   └── RunState        # Deterministic cache invalidation and artifact management
├── Sources
│   ├── Manifest        # Provenance registry, SHA-256 verification, data declarations
│   ├── Fetchers        # ETOPO, CHS, GLORYS, WOA23, Open-Meteo, TPXO9, Natural Earth
│   ├── RegionalCube    # 4D spatiotemporal oceanographic assimilation pipeline
│   ├── Catalog         # DatasetCatalog, DatasetEntry, search, metadata serialization
│   └── Lakehouse       # Medallion directory layout (bronze, silver, gold)
└── Partitioning
    ├── SpatialUnits    # Hexagonal, CVT (Lloyd's relaxation), KVT, QVT, BVT, AVT, lattice
    ├── Tessellation    # Voronoi diagrams, Delaunay triangulation, centroid relaxation
    ├── HexMesh         # Planar and geographic hexagonal packing and polygon generation
    ├── SpatialGraphs   # Adjacency matrices, KNN, radius graphs, island bridging
    ├── BYM2Scaling     # Spectral scaling factor tau calculation for intrinsic CAR models
    ├── Temporal        # Discretization into quantile, regular, Jenks, and k-means bins
    ├── CrossValidation # Spatial block cross-validation (k-means and grid blocking)
    ├── Resharding      # Linear cross-mesh transfer operators (area-weighted, kernel)
    └── HSIResharding   # Habitat Suitability Index spatial re-projection
├── Visualization & Mapping (TethysPlots.jl)
│   ├── Static Graphics # Publication-grade vector maps via Plots.jl
│   │   ├── Choropleth & Surfaces: choropleth, plot_hexagonal_field, plot_spatial_surface
│   │   ├── Graphs & Tessellations: spatial_graph_plot, plot_partition
│   │   ├── Hydrodynamics: plot_advection_arrows, plot_velocity_field, plot_hydrodynamic_section
│   │   └── Ecology & Tracking: plot_hsi_choropleth, plot_diffusion_map, timeseries_ci
│   └── Interactive GIS # Standalone web dashboards via Leaflet.js
│       ├── Vector Maps: leaflet_choropleth, leaflet_spatial_graph, leaflet_spacetime_map
│       ├── Currents & Flow: leaflet_advection_arrows, leaflet_current_density_map
│       └── Tracking: leaflet_particle_tracks, leaflet_hydrodynamic_dashboard
```

---

## Core Multidimensional Data Model

The data abstraction decouples physical array layouts from geospatial coordinates and storage formats.

### `Dimension`
Defines an axis of an N-dimensional array:
- `name::Symbol`: Axis identifier (e.g., `:lon`, `:lat`, `:time`, `:depth`).
- `size::Int`: Number of elements along the dimension.
- `coords::Any`: Ordered coordinate tick vector or range.
- `dim_type::DimensionType`: Classification as `DIM_SPATIAL`, `DIM_TEMPORAL`, `DIM_PARAMETRIC`, or `DIM_GENERIC`.
- `metadata::Dict{String, Any}`: Unit strings, calendar designations, and descriptions.

### `CoordinateSystem`
Carries coordinate system metadata:
- `crs::String`: EPSG code (e.g., `"EPSG:4326"`, `"EPSG:32620"`), WKT string, or PROJ string.
- `datum::String`: Geodetic reference datum (e.g., `"WGS84"`).
- `is_geographic::Bool`: Flags angular coordinates versus projected metric coordinates.

### `GeoArray{T, N}`
Wraps an $N$-dimensional array with metadata:
- `data::AbstractArray{T, N}`: Numerical tensor (in-memory `Array` or memory-mapped Zarr slice).
- `dims::Vector{Dimension}`: Vector of $N$ axis definitions matching `size(data)`.
- `crs::CoordinateSystem`: Coordinate reference system of spatial axes.
- `metadata::Dict{String, Any}`: Variable-level attributes (long name, standard units, missing values).

### `GeoDataset`
Multi-variable container corresponding to an oceanographic state:
- `variables::Dict{String, GeoArray}`: Named data arrays sharing or spanning coordinate subsets.
- `coords::Dict{Symbol, GeoArray}`: 1D coordinate vectors.
- `dims::Dict{Symbol, Dimension}`: Master dictionary of dimension specifications.
- `crs::CoordinateSystem`: Global dataset CRS.
- `metadata::Dict{String, Any}`: Global attributes (source URL, processing history, conventions).
- `backend::Union{GeoBackend, Nothing}`: Underlying persistence driver.
- `uri::String`: Storage location (URI or filesystem path).

```julia
using Tethys

# Constructing spatial dimensions
d_lon = Dimension(name = :lon, size = 100, coords = range(-65.0, -60.0, length = 100))
d_lat = Dimension(name = :lat, size = 80, coords = range(42.0, 46.0, length = 80))
d_time = Dimension(name = :time, size = 12, coords = 1:12, dim_type = DIM_TEMPORAL)

# Allocating GeoArray
temp_data = rand(Float32, 100, 80, 12)
temp_arr = GeoArray(
    temp_data;
    dims = [d_lon, d_lat, d_time],
    crs = CoordinateSystem(crs = "EPSG:4326"),
    metadata = Dict("units" => "degC", "long_name" => "Potential Temperature")
)

# Bundling into GeoDataset
ds = GeoDataset(
    Dict("temperature" => temp_arr),
    Dict(:lon => GeoArray(collect(d_lon.coords); dims = [d_lon]),
         :lat => GeoArray(collect(d_lat.coords); dims = [d_lat]),
         :time => GeoArray(collect(1:12); dims = [d_time])),
    Dict(:lon => d_lon, :lat => d_lat, :time => d_time)
)
```

---

## Storage Backends and Hierarchical Lakehouse

`Tethys.jl` persists spatial cubes using chunked and columnar storage technologies.

### Storage Backends
- **`:zarr`**: Compressed chunked N-dimensional directory store, ideal for 3D/4D spatial hydrodynamic simulations and raster data.
- **`:ncdatasets`**: NetCDF-4 / HDF5 and NCZarr driver via `NCDatasets.jl`.
- **`:geoparquet`**: Columnar table format via `Parquet2.jl` for point observations, biological survey tows, and Lagrangian particle trajectories.

### Medallion Lakehouse Hierarchy
Data assets are organized across three analytical tiers:
- **Bronze Tier (`lakehouse/bronze/`)**: Unaltered raw downloads (ETOPO NetCDF, WOA23 ASCII, TPXO9 binaries) preserving exact external provenance.
- **Silver Tier (`lakehouse/silver/`)**: Standardized, validated, and quality-controlled datasets reprojected to standard CRS and cleaned of non-finite values.
- **Gold Tier (`lakehouse/gold/`)**: Integrated model-ready data cubes, assimilated state spaces, and spatial graph discretization tables.

### `GeoStorage` Analytical API
Provides group-based variable reading and writing into lakehouse stores without loading full arrays into memory:

```julia
# Open hierarchical lakehouse store
gs = open_geostorage("lakehouse/gold/ocean_model.zarr"; backend = :zarr, create = true)

# Create group for simulation step
create_storage_group(gs, "step_001")

# Write variable slice
write_storage_variable!(gs, "step_001", "salinity", rand(Float32, 50, 50, 20))

# Introspect available groups and read back data
groups = list_storage_groups(gs)
sal = read_storage_variable(gs, "step_001", "salinity")

close_geostorage(gs)
```

---

## Coordinate Reference Systems and Projections

Geographic computations and numerical ocean models operate in metric Cartesian space. `Tethys.jl` wraps `CoordRefSystems.jl` to provide geodesy and coordinate transformations.

### Supported Projections
- **Geographic (`GeographicProjection`)**: Angular longitude/latitude on the WGS84 ellipsoid ($a = 6378137.0\,\text{m}$, $f = 1/298.257223563$).
- **Universal Transverse Mercator (`UTMProjection`)**: Conformal transverse cylindrical projection preserving local conformal geometry across $6^\circ$ longitudinal zones.
- **Lambert Conformal Conic (`LambertProjection`)**: Secant conical projection with two standard parallels ($\phi_1, \phi_2$), standard for regional mid-latitude oceanography.
- **Mercator (`MercatorProjection`)**: Conformal cylindrical projection maintaining rhumb lines.
- **Rotated Numerical Grid (`RotatedFrame`)**: Custom planar rotation for shelf-aligned oceanographic simulation domains.

### Mathematical Formulations

#### Lambert Conformal Conic Forward Transformation
Given central meridian $\lambda_0$, origin latitude $\phi_0$, and standard parallels $\phi_1, \phi_2$:
$$n = \frac{\ln(\cos\phi_1) - \ln(\cos\phi_2)}{\ln\left(\tan(\frac{\pi}{4} + \frac{\phi_2}{2})\right) - \ln\left(\tan(\frac{\pi}{4} + \frac{\phi_1}{2})\right)}$$
$$F = \frac{\cos\phi_1 \left(\tan(\frac{\pi}{4} + \frac{\phi_1}{2})\right)^n}{n}$$
$$\rho(\phi) = \frac{a F}{\left(\tan(\frac{\pi}{4} + \frac{\phi}{2})\right)^n}, \quad \theta = n(\lambda - \lambda_0)$$
$$x = \rho \sin\theta + x_0, \quad y = \rho_0 - \rho \cos\theta + y_0$$

#### Vector Rotation Between Geographic and Grid Frames
Let $\theta_{\text{rot}}$ be the grid alignment angle and $\gamma$ be the local grid convergence angle. For zonal and meridional velocities $(u_{\text{geo}}, v_{\text{geo}})$:
$$\alpha = \theta_{\text{rot}} + \gamma$$
$$\begin{bmatrix} u_{\text{grid}} \\ v_{\text{grid}} \end{bmatrix} = \begin{bmatrix} \cos\alpha & \sin\alpha \\ -\sin\alpha & \cos\alpha \end{bmatrix} \begin{bmatrix} u_{\text{geo}} \\ v_{\text{geo}} \end{bmatrix}$$

```julia
# Resolve CRS from EPSG identifier
proj_utm = build_projection("EPSG:32620")

# Forward projection (Lon/Lat -> Metric X/Y)
x, y = project_forward(proj_utm, -63.57, 44.65)

# Inverse projection (Metric X/Y -> Lon/Lat)
lon, lat = project_inverse(proj_utm, x, y)

# Rotate velocity vectors to shelf orientation
u_shelf, v_shelf = rotate_vector(1.2, -0.4, -30.0)
```

---

## Computational Geometry and Marine Domain Masking

Numerical hydrodynamic solvers and spatial assessment tools require accurate boundary delineation between terrestrial topography and ocean cells.

### Methods and Capabilities
- **`point_in_polygon(x, y, poly_x, poly_y)`**: Exact Jordan curve theorem ray-casting algorithm to test containment in arbitrary complex non-convex coastal polygons.
- **`is_marine_water(lon, lat; coastline)`**: Determines whether a geographic coordinate is located in open marine waters against high-resolution coastline polygons.
- **`buffer_distance_to_degrees(dist_km, lat)`**: Converts planar buffer distances in kilometers to local angular degrees accounting for latitude convergence ($\Delta \lambda = \Delta x / (111.32 \cos\phi)$).
- **`smooth_bathymetry(bathy, min_depth, max_depth)`**: Topographic relaxation and clamping. Constraining maximum water depth bounds the barotropic gravity wave speed:
$$c_{\text{grav}} = \sqrt{g H_{\max}}$$
Restricting $H_{\max} \le 1500\,\text{m}$ bounds $c_{\text{grav}} \le 121.3\,\text{m/s}$, ensuring numerical stability during split-explicit free-surface time subcycling.
- **Hypertidal Dampening**: Masking shallow dynamic estuaries (such as the upper Bay of Fundy) prevents non-linear tidal resonance and vertical velocity spikes in regional shelf models.

---

## Data Ingestion, Physical Fetchers, and Provenance Manifest

The `Manifest` module enforces reproducible data acquisition with strict metadata, URLs, and cryptographic hash verification.

### Supported Physical Sources
- **Bathymetry**:
  - `ETOPO 2022`: 15 arc-second global relief model (`fetch_erddap_bathymetry`).
  - `CHS`: Canadian Hydrographic Service non-navigational digital elevation models.
- **Ocean Hydrodynamics & Climatology**:
  - `GLORYS12V1`: Copernicus Marine $1/12^\circ$ daily reanalysis carrying 3D velocities, temperature, and salinity (`fetch_copernicus_physics_subset`).
  - `WOA23`: World Ocean Atlas 2023 decadal monthly hydrographic climatology (`fetch_woa23`, `load_woa23_interpolators`).
- **Atmospheric Forcing**:
  - `Open-Meteo`: Hourly 10-meter surface wind components ($u_{10}, v_{10}$) converted to kinematic surface wind stress via the Large & Pond (1981) drag formulation (`fetch_open_meteo_winds`).
- **Tidal Forcing**:
  - `TPXO9-Atlas`: High-resolution global tidal atlas supplying complex harmonic velocity constituents ($M_2, S_2, N_2, K_1, O_1$) for open-boundary condition generation (`fetch_tpxo9_velocity`).
- **Coastline Geography**:
  - `Natural Earth`: Global vector shorelines (`fetch_natural_earth_coastline`).
  - Bundled Scotian Shelf rings: Static, validated coastal contours for Atlantic Canada (`SCOTIAN_SHELF_COASTLINE`).

### Provenance and Verification
Every download is indexed in `DATA_SOURCES` and tracked via `data_provenance.json`:

```julia
# Fetch verified inputs with SHA-256 integrity checks
bathymetry_file = fetch_input(:bathymetry, "lakehouse/bronze"; strict = true)

# Inspect declared properties
manifest_entry = data_source(:bathymetry)
println("Source Title: ", manifest_entry.title)
println("License:      ", manifest_entry.license)
```

---

## Spatial and Spatiotemporal Partitioning Subsystem

In Bayesian spatiotemporal modeling, continuous Gaussian processes scale with cubic complexity $\mathcal{O}(N^3)$ in the number of observations $N$. Areal Gaussian Markov Random Field (GMRF) models (such as BYM2 and ICAR in `TethysBayes`), in contrast, scale as $\mathcal{O}(S^{1.5})$ to $\mathcal{O}(S)$, where $S$ is the number of discrete spatial units.

The `Tethys.Partitioning` subsystem discretizes continuous 2D spatial domains into discrete areal units $\Omega = \bigcup_{i=1}^S A_i$, constructs topological neighborhood adjacency matrices $W \in \mathbb{R}^{S \times S}$, enforces graph connectivity, scales precision matrices for BYM2 models, discretizes temporal dimensions, executes spatial cross-validation, and provides cross-mesh linear transfer operators.

```
                               ┌────────────────────────────────────────┐
                               │  Continuous Observations (s_x, s_y, t)  │
                               └───────────────────┬────────────────────┘
                                                   │
                   ┌───────────────────────────────┴───────────────────────────────┐
                   ▼                                                               ▼
     ┌────────────────────────────┐                                  ┌────────────────────────────┐
     │   Spatial Partitioning     │                                  │   Temporal Discretization  │
     │  (:avt, :cvt, :hex, etc.)  │                                  │  (Quantile, Jenks, KMeans) │
     └─────────────┬──────────────┘                                  └─────────────┬──────────────┘
                   │                                                               │
                   ▼                                                               ▼
     ┌────────────────────────────┐                                  ┌────────────────────────────┐
     │  Areal Units & Graph (W)   │                                  │     Time Units (t_idx)     │
     │  Centroids & Polygons      │                                  │     Breaks & Midpoints     │
     └─────────────┬──────────────┘                                  └─────────────┬──────────────┘
                   │                                                               │
                   └───────────────────────────────┬───────────────────────────────┘
                                                   │
                                                   ▼
                                 ┌───────────────────────────────────┐
                                 │  Spatiotemporal Indexing (st_idx) │
                                 │  BYM2 Spectral Scaling Factor (τ) │
                                 │  Spatial Block Cross-Validation   │
                                 │  Cross-Mesh Transfer Operator (P) │
                                 └───────────────────────────────────┘
```

### Mathematical Formulations and Tessellation Algorithms

#### Voronoi Tessellation Fundamentals
Given a bounded 2D domain $\Omega \subset \mathbb{R}^2$ and a set of $S$ distinct generator seeds $C = \{c_1, c_2, \dots, c_S\} \subset \Omega$, the Voronoi cell $V_i$ corresponding to generator $c_i$ is defined as:
$$V_i = \left\{ x \in \Omega \;\middle|\; \|x - c_i\|_2 \le \|x - c_j\|_2, \quad \forall j \ne i \right\}$$

The collection $\mathcal{V}(C) = \{V_1, V_2, \dots, V_S\}$ forms a convex polygonal partition of $\Omega$. The Delaunay triangulation $\mathcal{D}(C)$ is the planar geometric dual of $\mathcal{V}(C)$: two seeds $c_i$ and $c_j$ share an edge in $\mathcal{D}(C)$ if and only if their Voronoi cells share a boundary segment $\partial V_i \cap \partial V_j \ne \emptyset$. In `Tethys`, all Voronoi polygons are computed via Delaunay triangulation (`DelaunayTriangulation.jl`) and clipped against the domain's convex or buffered hull geometry using `LibGEOS.jl`.

#### Centroidal Voronoi Tessellation (`:cvt`)
A Centroidal Voronoi Tessellation (CVT) is a Voronoi diagram where each generator seed $c_i$ coincides with the geometric center of mass (centroid) $c_i^*$ of its associated Voronoi region $V_i$:
$$c_i^* = \frac{\int_{V_i} x \, \rho(x) \, dx}{\int_{V_i} \rho(x) \, dx}$$

When the density function $\rho(x) \equiv 1$ (uniform density), $c_i^*$ is the geometric centroid. CVTs minimize the spatial quantization energy:
$$\mathcal{E}(C, \mathcal{V}) = \sum_{i=1}^S \int_{V_i} \|x - c_i\|^2 \, dx$$

Lloyd's Relaxation algorithm executes the following sequence:
- Sample $S$ initial seeds $C^{(0)} = \{c_1^{(0)}, \dots, c_S^{(0)}\}$ using density-weighted KD-tree sampling.
- Construct $\mathcal{V}(C^{(t)})$ clipped to the domain boundary $\Omega$.
- Compute geometric centroids $c_i^{(t+1)} = \text{centroid}(V_i^{(t)})$.
- Compute mean displacement $\delta = \frac{1}{S} \sum_{i=1}^S \|c_i^{(t+1)} - c_i^{(t)}\|$ and terminate when $\delta < \text{tolerance}$ or upon reaching `max_iter`.

#### K-Means Density-Balanced Voronoi Tessellation (`:kvt`)
Unlike CVT which moves centroids toward the geometric center of the polygon, KVT moves seeds toward the empirical mean of the observational data points residing within each Voronoi cell:
$$c_i^{(t+1)} = (1 - \gamma) c_i^{(t)} + \gamma \left( \frac{1}{|N_i|} \sum_{x_j \in N_i} x_j \right)$$
where $N_i = \{x_j \in \text{Data} \mid \arg\min_k \|x_j - c_k\| = i\}$ is the set of points assigned to cell $i$, and $\gamma \in (0, 1]$ is a relaxation damping factor (initial $\gamma_0 = 0.7$, decaying by $0.99$ per iteration). This clusters smaller, high-resolution polygons in regions with dense observations while expanding larger polygons in sparse regions, keeping point counts per cell balanced.

#### Quadtree Voronoi Tessellation (`:qvt`)
Hierarchical space-partitioning method that recursively subdivides regions into four quadrants based on bivariate medians:
$$\tilde{x} = \text{median}(\{x_j \mid x_j \in R\}), \quad \tilde{y} = \text{median}(\{y_j \mid y_j \in R\})$$
Subdivision continues on regions exceeding `min_points` until the target unit count `target_units` or coefficient of variation (CV) threshold is achieved.

#### Binary Voronoi Tessellation (`:bvt`)
Recursively bisects regions along their principal axis of maximum variance:
$$d^* = \arg\max_{d \in \{x, y\}} \operatorname{Var}(p_d \mid p \in R)$$
The region is bisected along the median of dimension $d^*$:
$$R_{\text{left}} = \{p \in R \mid p_{d^*} \le \text{median}(p_{d^*})\}, \quad R_{\text{right}} = \{p \in R \mid p_{d^*} > \text{median}(p_{d^*})\}$$
This method is fast, robust to anisotropic coordinate scaling, and adapts to elongated spatial tracks or linear features (e.g., river networks, shelf breaks).

#### Hierarchical Voronoi Tessellation (`:hvt`)
Combines $k$-means spatial initialization with iterative Lloyd refinement and adaptive cell splitting. When a Voronoi cell contains more than `max_points` observations after centroid stabilization, the cell is split into two child seeds placed along its local dispersion axis:
$$c_{\text{new}, 1} = \bar{x}_{\text{cluster}} \times 0.99, \quad c_{\text{new}, 2} = \bar{x}_{\text{cluster}} \times 1.01$$

#### Agglomerative Voronoi Tessellation (`:avt`)
Bottom-up hierarchical clustering method. It begins in an over-partitioned state ($S_{\text{init}} = 2 \cdot \text{target\_units}$) and iteratively merges the smallest, underpopulated, or constraint-violating polygon into its nearest neighbor:
$$c_{\text{merged}} = \frac{n_i c_i + n_j c_j}{n_i + n_j}$$
where $j = \arg\min_{k \ne i} \|c_i - c_k\|^2$. Merging continues until all remaining units satisfy `min_points`, `min_time_slices`, and `min_area` constraints.

#### Hexagonal Grid Tessellation (`:hexagonal` / `:hexbin`)
Hexagonal tessellations partition 2D space into regular honeycombs. Hexagons have optimal geometric properties:
- **Maximum Area-to-Perimeter Ratio**: Minimizes boundary edge distortion.
- **Isotropic Neighborhood**: Every interior hexagon shares equidistant boundaries with exactly 6 neighbors, eliminating the diagonal vs. orthogonal distance asymmetry of square grids.

```
           (x, y + R)
          /          \
  (x - dx/2, y + R/2) (x + dx/2, y + R/2)
        |              |
        |   (x, y)     |  Radius: R
        |              |  Width:  dx = √3 · R
  (x - dx/2, y - R/2) (x + dx/2, y - R/2)
          \          /    Height: dy = 1.5 · R
           (x, y - R)
```

Geometry and formulas for regular hexagons with radius $R$ (center to vertex):
- Area: $A_{\text{hex}} = \frac{3\sqrt{3}}{2} R^2 \approx 2.598 R^2$
- Target Radius from Target Area: $R = \sqrt{\frac{2 A_{\text{target}}}{3\sqrt{3}}}$
- Horizontal column spacing: $\Delta x = \sqrt{3} R$
- Vertical row spacing: $\Delta y = \frac{3}{2} R$
- Row Offset: Odd rows are shifted horizontally by $\frac{\sqrt{3}}{2} R$.
- Vertices for cell centered at $(x_c, y_c)$:
$$v_k = \left( x_c + R \cos\left(\frac{\pi}{6} + \frac{k\pi}{3}\right), \; y_c + R \sin\left(\frac{\pi}{6} + \frac{k\pi}{3}\right) \right), \quad k = 0, \dots, 5$$

#### Regular Lattice Grid (`:lattice`)
Square or rectangular raster grid partitioning. Given bounding box $[x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]$:
- Cell width $L_x = \frac{x_{\max} - x_{\min}}{\text{cols}}$
- Cell height $L_y = L_x \times \text{aspect\_ratio}$
`Tethys` implements analytical $\mathcal{O}(N)$ stencil generation for grid adjacency matrices without geometric intersection overhead.

#### Force-Directed Inferred Spatial Layout (`assign_spatial_units_inferred`)
When observational data contains an adjacency matrix $W$ but lacks geographic coordinates $(s_x, s_y)$ (such as standard regional benchmark datasets), `assign_spatial_units_inferred` reconstructs continuous coordinates via a force-directed spring layout:
$$c_i^{(t+1)} = c_i^{(t)} + \eta \left( \frac{1}{|N(i)|} \sum_{j \in N(i)} c_j^{(t)} - c_i^{(t)} \right)$$
where $\eta$ is the learning rate. Once coordinates stabilize, Voronoi boundaries and a hull geometry are constructed.

---

### Spatial Topology, Adjacency Graphs, and BYM2 Scaling

#### Contiguity Types: Queen vs. Rook
Spatial adjacency matrices $W \in \{0, 1\}^{S \times S}$ define the neighborhood topology:
- **Queen Contiguity**: Areas $A_i$ and $A_j$ are neighbors ($W_{ij} = 1$) if they share at least one boundary point or edge ($\partial A_i \cap \partial A_j \ne \emptyset$).
- **Rook Contiguity**: Areas $A_i$ and $A_j$ are neighbors if they share a boundary linear edge of non-zero length ($\operatorname{length}(\partial A_i \cap \partial A_j) > 0$).

#### Disconnected Components and Island Bridging (`ensure_connected!`)
Spatial models (such as ICAR or BYM2) require a connected spatial graph $\mathcal{G}$ to ensure an identifiable singular precision matrix with a single rank-deficiency of 1. If physical geography (e.g., islands, isolated banks) creates $K > 1$ disjoint components:
- Calculate the geometric center of each connected component.
- Use a 2D KD-Tree to search for the closest pair of components.
- Connect the closest node pair $(u, v)$ across the two components with a mutual edge:
$$(u^*, v^*) = \arg\min_{u \in C_a, v \in C_b} \|c_u - c_v\|_2^2$$
This guarantees a connected graph with minimal perturbation to the graph topology.

#### BYM2 Spectral Scaling Factor ($\tau$)
The BYM2 model (Riebler et al., 2016) decomposes spatial random effects into an unstructured white-noise effect ($v$) and a structured intrinsic GMRF ($u$):
$$\theta_s = \sigma_s \left( \sqrt{1 - \rho} \cdot v_s + \sqrt{\rho / \tau} \cdot u_s \right)$$
where $u \sim \operatorname{ICAR}(W)$ with singular precision matrix $Q = \operatorname{diag}(W \mathbf{1}) - W$. To place the structured component on the same variance scale as the unstructured component (making $\rho \in [0, 1]$ interpretable as the proportion of variance explained by spatial structure), the scaling factor $\tau$ is computed from the non-zero eigenvalues $\lambda_1, \dots, \lambda_{S-1}$ of $Q$:
$$\tau = \exp\left( -\frac{1}{S-1} \sum_{i=1}^{S-1} \ln \lambda_i \right)$$
In `Tethys`, this is computed via `scaling_factor_bym2(W)`:
```julia
tau = scaling_factor_bym2(W)
```

#### Spatial Weighting Transformations (`spatial_weights_matrix`)
Standardized spatial econometric weighting schemes via `spatial_weights_matrix(W; style=...)`:
- **`:binary`**: Unweighted symmetric adjacency $W_{ij} \in \{0, 1\}$ for ICAR and BYM2 priors.
- **`:row_standardized`**: Row-normalized $W_{ij}^* = W_{ij} / \sum_k W_{ik}$ where row sums equal 1, representing spatial moving averages in SAR/SEM models ($W y$).
- **`:variance_stabilized`**: $W_{ij}^* = W_{ij} / \sqrt{\sum_k W_{ik}}$ to stabilize variance across units with unequal neighbor counts.

---

### Partitioning Options and Return Value Specification

`assign_spatial_units` provides granular controls over polygon dimensions, unit counts, and convergence thresholds:

| Option | Type | Default | Relevant Methods | Statistical & Mathematical Implications |
| :--- | :--- | :--- | :--- | :--- |
| `area_method` | `Symbol` | `:avt` | All | Selects algorithm (`:avt`, `:cvt`, `:kvt`, `:qvt`, `:bvt`, `:hvt`, `:hexagonal`, `:lattice`). Determines whether partitioning is geometric, density-adaptive, or regular grid. |
| `target_units` | `Integer` | `10` | All | Desired number of spatial partitions $S$. Controls model resolution: larger $S$ captures fine-scale heterogeneity but increases parameter count and computation. |
| `exact_units` | `Bool` | `false` | All | When `true`, strictly guarantees *exactly* `target_units` final polygons via deterministic post-clipping bisection (if $K < S$) or adjacent-pair union merging (if $K > S$). |
| `target_area` | `Real` | `nothing` | All | Desired mean area per polygon in coordinate units. Automatically computes `target_units = max(1, round(total_area / target_area))` and scales hexagon radius $R = \sqrt{\frac{2 A}{3\sqrt{3}}}$ or lattice side length $L = \sqrt{A}$. |
| `min_area` | `Real` | `0.0` | All | Minimum allowed polygon surface area. Prevents tiny boundary fragments/slivers generated during polygon clipping from skewing spatial statistics. |
| `max_area` | `Real` | `Inf` | All | Maximum allowed polygon surface area. In hierarchical methods, forces subdivision of large rural/unpopulated cells. |
| `min_points` | `Integer` | `1` | `:avt`, `:kvt`, `:qvt`, `:bvt`, `:hvt` | Minimum observations per polygon. Setting `min_points > 1` prevents unobserved or sparse units that produce poorly identified random intercepts. |
| `max_points` | `Integer` | `length(s_x)` | `:qvt`, `:bvt`, `:hvt` | Maximum observations per polygon. In tree methods, triggers child splitting when a cluster exceeds this threshold. |
| `min_time_slices` | `Integer` | `1` | `:avt`, `:kvt`, `:bvt` | Minimum unique time points required in each polygon. Essential for spatiotemporal models so each spatial unit has time series depth. |
| `t_idx` | `Vector{Int}` | `ones(...)` | `:avt`, `:kvt`, `:bvt` | Vector of observation time stamps passed to spatial partitioning to evaluate `min_time_slices`. |
| `tolerance` | `Real` | `0.1` | `:cvt`, `:kvt`, `:avt`, `:hvt` | Centroid displacement tolerance $\delta$. Smaller values produce stricter geometric convergence at the cost of additional iterations. |
| `buffer_dist` | `Real` | `0.5` | Voronoi methods | Distance to expand the convex hull boundary. Ensures peripheral data points do not lie outside clipped polygon boundaries. |
| `merge_small_polygons` | `Bool` | `false` | All | Automatically merges polygons with $\text{area} < \text{min\_area}$ into their neighbor sharing the longest boundary edge using `LibGEOS.union`. |
| `prune_empty` | `Bool` | `false` | All | Drops spatial units containing 0 data points and updates the graph and $W$ matrix. Reduces model dimension to active units only. |
| `radius` | `Real` | `nothing` | `:hexagonal` | Explicit hexagon radius $R$. Direct control over honeycomb cell resolution. |
| `length_scale` | `Real` | `nothing` | `:lattice` | Explicit square cell side length $L$. Direct control over raster cell resolution. |
| `grid_resolution` | `Int` or `(Int, Int)` | `nothing` | `:lattice` | Explicit $(rows, cols)$ or $N \times N$ tiling for regular grids. |
| `aspect_ratio` | `Real` | `1.0` | `:lattice` | Cell aspect ratio $L_y / L_x$. Accommodates non-square geographic bounding boxes without cell distortion. |
| `input_polygons` | `Vector` | `nothing` | Custom | User-supplied custom geometries (e.g. Shapefile, GeoJSON). Bypasses automatic tessellation while constructing centroids, boundaries, and graph topology. |
| `geom_hull` | `LibGEOS.Geometry` | `nothing` | All | Custom boundary clipping polygon (e.g. administrative boundary, coastline). |

#### Return Value Structure
The output of `assign_spatial_units` is a unified `NamedTuple`:
- `centroids`: `Vector{Tuple{Float64, Float64}}` of unit centroids.
- `polygons`: `Vector{Vector{Tuple{Float64, Float64}}}` of closed polygon boundary vertices.
- `adjacency_edges`: `Vector{Tuple{Tuple, Tuple}}` of centroid pairs sharing an edge.
- `graph`: `Graphs.SimpleGraph` spatial topology object.
- `W`: $S \times S$ sparse adjacency matrix.
- `hull_coords`: Bounding envelope / clipping hull coordinates.
- `s_idx`: `Vector{Int}` (length $N$) mapping observations to unit index $1:S$.
- `s_x`, `s_y`: Original coordinate vectors.
- `s_vals`: Vector of unique spatial unit labels ($1:S$).
- `areas`: `Vector{Float64}` of individual polygon surface areas.
- `point_counts`: `Vector{Int}` of observation counts per polygon.
- `n_units`: Total number of spatial units $S$.
- `metrics`: `NamedTuple` containing `mean_density`, `sd_density`, `cv_density`, `min_density`, `max_density`, `mean_area`, `sd_area`, `cv_area`, `min_area`, `max_area`, and `total_area`.
- `termination_reason`: Diagnostic string indicating algorithm stopping condition.

---

### Temporal and Spatiotemporal Discretization

#### Discretization Methods (`discretize_data`)
The `discretize_data` function partitions continuous 1D variables (such as time $t$, covariates, or coordinates) into $K$ categorical intervals:
- **`"quantile"`**: Bins defined by empirical quantiles $q_k = F_X^{-1}(k/K)$. Guarantees balanced observation counts per bin; robust against heavy-tailed data.
- **`"regular"`**: Equal-width bins $\Delta x = \frac{\max(X) - \min(X)}{K}$. Standard uniform time intervals (e.g., calendar years, fixed time steps).
- **`"quantile_regular"`**: Equal-width bins between quantiles $[\alpha, 1-\alpha]$ (default: $[0.025, 0.975]$). Prevents extreme temporal outliers from over-extending outer bin widths.
- **`"kmeans"`**: 1D $k$-means clustering minimizing within-bin variance $\sum_k \sum_{x \in B_k} (x - \mu_k)^2$. Natural clustering for irregularly sampled temporal data.
- **`"jenks"`**: Jenks Natural Breaks optimization maximizing Goodness of Variance Fit (GVF). Optimal classification for multimodal distributions with distinct regimes.
- **`"provided"`**: Explicit user-supplied break points `brks`. Predefined regulatory, seasonal, or administrative thresholds.

#### Unified Spatiotemporal Indexing (`assign_spatiotemporal_units`)
For spatiotemporal models with spatiotemporal interaction terms (e.g. Type I–IV interaction models, Knorr-Held 2000), `assign_spatiotemporal_units` synchronizes spatial areal unit indices $s \in \{1, \dots, S\}$ and temporal intervals $t \in \{1, \dots, T\}$ into a single joint index $st \in \{1, \dots, S \cdot T\}$:
$$st = (t - 1) \cdot S + s$$

```julia
st_res = assign_spatiotemporal_units(df; 
    space_x = :lon, 
    space_y = :lat, 
    time_var = :year, 
    area_method = :hexagonal, 
    target_units = 20, 
    time_method = "unique"
)
```
Returns:
- `au_spatial`: Spatial areal units object.
- `au_temporal`: Temporal units object.
- `s_idx`, `t_idx`, `st_idx`: Vectors of assignments for each observation.
- `S`, `T`, `ST`: Dimensional scalars ($S$, $T$, $S \times T$).
- `W_spatial`: Spatial adjacency matrix.
- `scaling_factor_spatial`: BYM2 precision scaling factor ($\tau$).

#### Spatial Block Cross-Validation (`spatial_block_cv`)
Random $k$-fold cross-validation produces overoptimistic predictive performance on spatial data due to spatial autocorrelation between adjacent training and test points (Roberts et al., 2017). `spatial_block_cv` partitions the domain into $K$ spatially disjoint blocks:
- **`:kmeans`**: Partitions coordinate space via $k$-means clustering into $K$ compact geographical folds.
- **`:grid`**: Partitions the bounding box into a regular grid and assigns checkerboard fold IDs.

```julia
folds = spatial_block_cv(df.lon, df.lat; n_folds = 5, method = :kmeans)
```

---

### Cross-Mesh Topological Resharding and Spatial Transfer Operators

In multi-tiered workflows, data sources at different physical layers often possess disparate spatial densities and resolutions. For example, sediment grab stations may form an areal graph $G_{\text{src}} = (V_{\text{src}}, W_{\text{src}})$ with $N_{\text{src}} = 25$ units, whereas final population assessments require predictions on a fine-scale management mesh $G_{\text{dest}} = (V_{\text{dest}}, W_{\text{dest}})$ with $N_{\text{dest}} = 80$ units.

`Tethys` provides linear transfer operators $P \in \mathbb{R}^{N_{\text{dest}} \times N_{\text{src}}}$ to map spatial fields seamlessly between irregular polygonal tessellations:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        CROSS-MESH GEOMETRIC TRANSFER OPERATOR                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  Source Mesh A (A_1, ..., A_m) ──► Latent Field u_src or Samples U_src [N_src × S]     │
│                                           │                                            │
│                 Transfer Matrix P: P_ji = Area(B_j ∩ A_i) / Area(B_j)                  │
│                                           │                                            │
│  Destination Mesh B (B_1, ..., B_n) ◄─────┴── u_dest = P × u_src                       │
│                                               U_dest = P × U_src                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Transfer Formulations
- **Area-Weighted Polygon Intersection (`method = :area_weighted`)**:
  For destination polygon $B_j$ and source polygon $A_i$, the transfer weight represents the fraction of $B_j$'s area overlapping $A_i$:
  $$P_{j, i} = \frac{\text{Area}(B_j \cap A_i)}{\text{Area}(B_j)}, \quad \sum_{i=1}^{N_{\text{src}}} P_{j, i} = 1$$
  Computed geometrically via `LibGEOS.intersection` across all intersecting polygon pairs.
- **Gaussian Kernel Spatial Decay (`method = :gaussian_kernel`)**:
  Transfer weights decay smoothly as a Gaussian function of Euclidean distance between unit centroids:
  $$\tilde{P}_{j, i} = \exp\left(-\frac{\|\mathbf{c}_{\text{dest}, j} - \mathbf{c}_{\text{src}, i}\|^2}{2 h^2}\right), \quad P_{j, i} = \frac{\tilde{P}_{j, i}}{\sum_{k=1}^{N_{\text{src}}} \tilde{P}_{j, k}}$$
  where $h$ is the spatial bandwidth parameter (defaults to mean nearest-neighbor distance).
- **Inverse Distance Weighting (`method = :inverse_distance`)**:
  $$\tilde{P}_{j, i} = \frac{1}{(\|\mathbf{c}_{\text{dest}, j} - \mathbf{c}_{\text{src}, i}\| + \epsilon)^p}, \quad P_{j, i} = \frac{\tilde{P}_{j, i}}{\sum_{k=1}^{N_{\text{src}}} \tilde{P}_{j, k}}$$

#### Sample Matrix vs. Moment Resharding
- **Full Monte Carlo Mode (`mode = :samples`)**: When downstream likelihoods are non-Gaussian (e.g., Gamma, Negative Binomial, Hurdle), resharding the raw $N_{\text{src}} \times S$ posterior sample matrix preserves higher-order moments, skewness, and multi-modal distributions:
$$\mathbf{U}_{\text{dest}} = \mathbf{P} \mathbf{U}_{\text{src}} \in \mathbb{R}^{N_{\text{dest}} \times S}$$
- **Moments Mode (`mode = :moments`)**: Reshards the posterior mean vector and variance vector directly:
$$\mathbf{u}_{\text{dest}} = \mathbf{P} \mathbf{u}_{\text{src}}, \quad \boldsymbol{\sigma}^2_{\text{dest}} = \mathbf{P} \boldsymbol{\sigma}^2_{\text{src}}$$

---

### Modeling Considerations and Best Practices

- **Projected Coordinate Reference Systems (CRS)**: All geometric calculations (Euclidean distance, Shoelace polygon area) assume planar Cartesian coordinates (e.g., UTM or local projected meters/kilometers). Longitude/latitude degrees should be projected to a suitable planar CRS prior to partitioning to avoid high-latitude distortion.
- **Modifiable Areal Unit Problem (MAUP)**: Aggregating point data into discrete polygons involves scale and zoning effects (Openshaw, 1984; Fotheringham & Wong, 1991). Sensitivity analysis should be performed by fitting models across multiple resolutions (`target_units = 15`, `target_units = 30`, `target_units = 60`) or comparing `:hexagonal` vs. `:cvt`.
- **Graph Connectivity**: BYM2 and ICAR spatial random effects require connected spatial graphs. `ensure_connected!` automatically bridges disconnected islands to prevent precision matrix singularity.
- **Computational Complexity**: Point assignment and Lloyd relaxation use `NearestNeighbors.KDTree`, scaling at $\mathcal{O}(N \log S)$. Tessellation of up to $10^6$ points across hundreds of areal units executes in seconds.

---

## Dataset Manipulation and Query API

`Tethys.jl` exposes functional verbs for operating on geospatial data cubes:

### API Functions
- **`load_dataset(path; backend)`**: Opens a `GeoDataset` from disk or lakehouse storage.
- **`save_dataset(path, ds; backend, overwrite)`**: Persists a `GeoDataset` to the chosen storage backend.
- **`slice_dataset(ds; lon, lat, time, depth)`**: Extracts orthogonal hyperslabs bounded by coordinate intervals.
- **`sample_dataset(ds, variables; lon, lat, time)`**: Samples values at point coordinates using bilinear spatial interpolation.
- **`regrid_dataset(ds, target_grid)`**: Remaps fields onto target spatial coordinates using bilinear or conservative remapping.
- **`select_dataset(ds, vars)`**: Subsets variables into a new lightweight dataset container.
- **`join_datasets(ds1, ds2; on)`**: Merges two datasets along shared coordinate dimensions.
- **`aggregate_dataset(ds, dim; func)`**: Reduces data cubes along a specified axis using summary statistics (`mean`, `sum`, `maximum`).

```julia
# Open analytical cube
ds = load_dataset("lakehouse/silver/regional_cube.zarr"; backend = :zarr)

# Slice geographical bounding box
shelf_subset = slice_dataset(ds; lon = (-64.0, -58.0), lat = (43.0, 47.0))

# Sample environmental variables at station coordinates
obs = sample_dataset(shelf_subset, ["temperature", "salinity"]; lon = -62.5, lat = 44.2)
```

---

## Metadata Catalog and Run State Management

### `DatasetCatalog`
Provides a queryable catalog registry for spatial assets:
- `register_dataset!(catalog, entry)`: Registers a dataset with spatial bounds, timestamps, and schema definitions.
- `find_datasets(catalog; name, bbox, time_range)`: Searches registered assets matching spatial and temporal criteria.
- `export_catalog_to_csv(catalog, file)`: Exports catalog inventory for external reporting.

### Deterministic Run State Invalidation (`purge_run_state`)
Enforces repeatable modeling pipelines by providing controlled artifact purging:
- Targets: `:ocean`, `:larval`, `:mesh`, `:bathymetry`, `:cube`, `:all`.
- Ensures intermediate cache artifacts are removed without deleting raw bronze inputs.

---

## Complete End-to-End Workflow Examples

### Example 1: Spatial Mesh Construction and Lakehouse Persistence
The following script loads bathymetric observations, builds an adaptive hexagonal discretization mesh, generates the neighborhood topology, computes the BYM2 spectral scaling factor, and stores the processed spatial structure in the lakehouse:

```julia
using Tethys
using DataFrames

# Define study region
study_bbox = (-65.5, -59.5, 42.0, 46.5)

# Generate regular hexagonal discretization mesh
println("Constructing hexagonal spatial mesh...")
mesh = build_hex_mesh(study_bbox; cell_size_km = 30.0)
n_units = length(mesh.centroids_lonlat)
println("Generated $(n_units) hexagonal areal units.")

# Construct topological spatial adjacency graph
W_sparse = spatial_weights_matrix(mesh.centroids_lonlat; method = :knn, k = 6)

# Compute BYM2 spectral scaling factor for spatial priors
tau_scale = scaling_factor_bym2(W_sparse)
println("Computed BYM2 spectral scaling factor tau = $(round(tau_scale, digits=4))")

# Package into GeoDataset
d_units = Dimension(name = :unit, size = n_units, coords = 1:n_units, dim_type = DIM_SPATIAL)
centroids_x = [pt[1] for pt in mesh.centroids_lonlat]
centroids_y = [pt[2] for pt in mesh.centroids_lonlat]

ds_mesh = GeoDataset(
    Dict("centroid_x" => GeoArray(centroids_x; dims = [d_units]),
         "centroid_y" => GeoArray(centroids_y; dims = [d_units])),
    Dict(:unit => GeoArray(collect(1:n_units); dims = [d_units])),
    Dict(:unit => d_units),
    CoordinateSystem(crs = "EPSG:4326"),
    Dict{String, Any}("tau" => tau_scale, "cell_size_km" => 30.0)
)

# Save processed mesh to lakehouse silver tier
output_path = "lakehouse/silver/spatial_mesh_30km.zarr"
save_dataset(output_path, ds_mesh; backend = :zarr, overwrite = true)
println("Successfully persisted spatial mesh to $(output_path)")
```

### Example 2: Cross-Mesh Topological Transfer of Latent Fields
The following script demonstrates computing the linear transfer operator $P$ to map posterior simulations from a coarse sampling mesh to a fine assessment mesh:

```julia
using Tethys

# Construct coarse and fine spatial partitionings
au_coarse = assign_spatial_units(df_coarse.lon, df_coarse.lat;
    area_method = :cvt,
    target_units = 20
)

au_fine = assign_spatial_units(df_fine.lon, df_fine.lat;
    area_method = :hexagonal,
    target_units = 60
)

# Compute the geometric transfer operator matrix P [N_dest x N_src]
P = compute_network_transfer_matrix(au_coarse, au_fine; method = :area_weighted)

# Reshard posterior sample matrix (20 units x 500 draws -> 60 units x 500 draws)
coarse_draws = randn(20, 500)
fine_results = reshard_spatial_field(coarse_draws, au_coarse, au_fine; mode = :samples)

println("Resharded fine-mesh posterior dimensions: ", size(fine_results.samples))
```

### Example 3: Visualization of Discretizations and Adjacency Graphs via TethysPlots
The following script illustrates publication-grade static visualization and standalone interactive Leaflet GIS dashboard creation using `TethysPlots.jl`:

```julia
using Tethys
using TethysPlots

# Discretize observations into regular hexagonal areal units
au = assign_spatial_units(obs_df;
    x = :lon,
    y = :lat,
    area_method = :hexagonal,
    target_units = 25,
    exact_units = true
)

# 1. Publication-grade static choropleth
p_static = plot_partition(au;
    data = obs_df,
    x = :lon,
    y = :lat,
    title = "Scotian Shelf Hexagonal Lattice"
)

# 2. Spatial graph adjacency network
W = spatial_weights_matrix(au.centroids; method = :knn, k = 4)
p_graph = spatial_graph_plot(au.polygons, W;
    centroids = au.centroids,
    title = "Topological Neighbor Network"
)

# 3. Interactive Leaflet GIS dashboard
m_interactive = leaflet_choropleth(au.polygons, [c[1] for c in au.centroids];
    title = "Interactive Hexagonal Dashboard",
    colorbar_label = "Longitude Centroid",
    cmap = :viridis
)
save_html(m_interactive, "shelf_interactive.html")
```

---

## Methodological References

- **Besag, J. (1974)**. Spatial interaction and the statistical analysis of lattice systems. *Journal of the Royal Statistical Society: Series B (Methodological)*, 36(2), 192–225.
- **Besag, J., York, J., & Mollié, A. (1991)**. Bayesian image restoration, with two applications in spatial statistics. *Annals of the Institute of Statistical Mathematics*, 43(1), 1–20.
- **Du, Q., Faber, V., & Gunzburger, M. (1999)**. Centroidal Voronoi tessellations: applications and algorithms. *SIAM Review*, 41(4), 637–676.
- **Fotheringham, A. S., & Wong, D. W. (1991)**. The modifiable areal unit problem in multivariate statistical analysis. *Environment and Planning A*, 23(7), 1025–1044.
- **Jenks, G. F. (1967)**. The data model concept in statistical mapping. *International Yearbook of Cartography*, 7, 186–190.
- **Knorr-Held, L. (2000)**. Bayesian modelling of inseparable space-time variation in disease rates. *Statistics in Medicine*, 19(17-18), 2555–2567.
- **Large, W. G., & Pond, S. (1981)**. Open ocean momentum flux measurements in moderate to strong winds. *Journal of Physical Oceanography*, 11(3), 324–336.
- **Lloyd, S. (1982)**. Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129–137.
- **Okabe, A., Boots, B., Sugihara, K., & Chiu, S. N. (2000)**. *Spatial Tessellations: Concepts and Applications of Voronoi Diagrams* (2nd ed.). John Wiley & Sons.
- **Openshaw, S. (1984)**. *The Modifiable Areal Unit Problem*. Geo Books, Norwich.
- **Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H. (2016)**. An intuitive Bayesian spatial model for disease mapping that accounts for scaling. *Statistical Methods in Medical Research*, 25(4), 1145–1165.
- **Roberts, D. R., et al. (2017)**. Cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure. *Ecography*, 40(8), 913–929.
- **Snyder, J. P. (1987)**. *Map projections—A working manual* (U.S. Geological Survey Professional Paper 1395). Washington, DC: U.S. Government Printing Office.
