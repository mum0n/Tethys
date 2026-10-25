### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity.
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
begin
    using Pkg
    project_root = isfile(joinpath(@__DIR__, "Project.toml")) ? @__DIR__ : joinpath(@__DIR__, "..")
    Pkg.activate(project_root)
    tethysplots_dir = normpath(joinpath(project_root, "..", "TethysPlots"))
    if isdir(tethysplots_dir) && !(tethysplots_dir in LOAD_PATH)
        push!(LOAD_PATH, tethysplots_dir)
    end
    using Tethys
    using TethysPlots
    using DataFrames
    using Random
    using Statistics
    using LinearAlgebra
    using SparseArrays
    using Plots
    default(size=(750, 480), palette=:viridis, legend=false)
    "Environment active: Tethys, TethysPlots, and Plots loaded."
end

# ╔═╡ 00000000-0000-0000-0000-000000000002
md"""
# Tethys — Multidimensional Oceanographic Data, Geodesy, and Spatial Partitioning

This interactive notebook demonstrates the foundational workflows of `Tethys.jl` and its visualization library `TethysPlots.jl`:

- **Multidimensional GeoDataset Architecture**: Axis definitions (`Dimension`), coordinate metadata (`CoordinateSystem`), and dataset wrapping (`GeoArray`, `GeoDataset`).
- **Spatial Slicing & Vertical Water-Column Profiling**: Bounding-box subsets and point sampling through continuous coordinate fields.
- **Geodesy & Rotated Oceanographic Velocity Frames**: Forward/inverse map projections (`CoordRefSystems.jl`) and vector transformations to shelf-aligned numerical grid orientations.
- **Coastal Geometry & Marine Domain Masking**: Topographic ray-casting, coastline containment, and barotropic wave speed capping.
- **Spatial Partitioning & Tessellation Algorithms**: Comparative analysis of hexagonal honeycombs (`:hexagonal`), Centroidal Voronoi (`:cvt`), and K-Means density balancing (`:kvt`).
- **Spatial Graph Topology & BYM2 Spectral Scaling**: Neighborhood adjacency matrices, island bridging via KD-tree links (`ensure_connected!`), and spectral scaling factor ($\tau$).
- **Spatiotemporal Synchronization**: Synchronizing space and time intervals into a unified index ($st = (t-1)S + s$).
- **Cross-Mesh Topological Transfer Operators**: Linear transfer matrices ($P_{ji}$) for mapping latent fields between irregular polygonal meshes.
- **Hierarchical Lakehouse Storage**: Low-level chunked variable reading and writing with `GeoStorage`.
- **Ecosystem Visualization (`TethysPlots.jl`)**: Publication-grade static vector choropleths, spatial graph topologies, hydrodynamic flow vectors, and standalone interactive Leaflet GIS dashboards.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000003
md"""
## 1. Multidimensional GeoDataset and 2D Spatial Heatmaps

The `GeoDataset` couples numerical tensors with named coordinate axes (`Dimension`) and coordinate reference systems (`CoordinateSystem`). Below, a synthetic 2D bottom temperature field is generated over the Scotian Shelf region ($\text{lon} \in [-66^\circ, -58^\circ]$, $\text{lat} \in [42^\circ, 47^\circ]$).
"""

# ╔═╡ 00000000-0000-0000-0000-000000000004
begin
    lons = collect(range(-66.0, -58.0, length = 50))
    lats = collect(range(42.0, 47.0, length = 40))
    d_lon = Dimension(name = :lon, size = length(lons), coords = lons, dim_type = DIM_SPATIAL)
    d_lat = Dimension(name = :lat, size = length(lats), coords = lats, dim_type = DIM_SPATIAL)

    # Synthetic bottom temperature field with regional spatial gradient
    temp_field = [
        Float32(4.0 + 3.0 * sin((x + 66.0) / 8.0 * π) - 2.5 * ((y - 42.0) / 5.0) + 0.3 * randn())
        for x in lons, y in lats
    ]

    crs_wgs84 = CoordinateSystem(crs = "EPSG:4326")
    ga_temp = GeoArray(temp_field; dims = [d_lon, d_lat], crs = crs_wgs84)
    ds_ocean = GeoDataset(
        Dict("temperature" => ga_temp),
        Dict(:lon => GeoArray(lons; dims = [d_lon]), :lat => GeoArray(lats; dims = [d_lat])),
        Dict(:lon => d_lon, :lat => d_lat)
    )

    # Visualizing 2D bottom temperature field
    heatmap(
        lons, lats, temp_field',
        title = "Bottom Temperature Field (°C)",
        xlabel = "Longitude (°E)",
        ylabel = "Latitude (°N)",
        colorbar = true
    )
end

# ╔═╡ 00000000-0000-0000-0000-000000000005
md"""
## 2. Spatial Slicing and Vertical Water-Column Profiling

`slice_dataset` extracts coordinate hyperslabs, while `sample_dataset` queries continuous fields at arbitrary station points. Below, a 3D ocean cube is generated to extract and plot a vertical temperature profile.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000006
begin
    depths = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    d_depth = Dimension(name = :depth, size = length(depths), coords = depths, dim_type = DIM_PARAMETRIC)

    # 3D temperature cube (lon × lat × depth) with exponential thermocline
    temp_3d = zeros(Float32, length(lons), length(lats), length(depths))
    for (k, z) in enumerate(depths)
        # Thermocline decay with depth: T(z) = T_bot + (T_surf - T_bot) * exp(-z / 100)
        decay = exp(-z / 120.0)
        temp_3d[:, :, k] = temp_field .* Float32(decay) .+ Float32(2.0 * (1.0 - decay))
    end

    ga_3d = GeoArray(temp_3d; dims = [d_lon, d_lat, d_depth], crs = crs_wgs84)
    ds_cube = GeoDataset(Dict("temperature" => ga_3d), Dict(:lon => ga_temp, :lat => ga_temp))

    # Continuous station profile query via sample_dataset
    stn_lon, stn_lat = -63.5, 44.2
    profile_pt = [
        sample_dataset(ds_cube, "temperature", Dict(:lon => stn_lon, :lat => stn_lat, :depth => z))
        for z in depths
    ]

    # Plot water column stratification profile
    plot(
        profile_pt, depths,
        yflip = true,
        title = "Water Column Temperature Profile at (-63.5°E, 44.2°N)",
        xlabel = "Temperature (°C)",
        ylabel = "Depth (m)",
        lw = 2.5,
        marker = :circle,
        color = :crimson
    )
end

# ╔═╡ 00000000-0000-0000-0000-000000000007
md"""
## 3. Geodesy and Rotated Oceanographic Velocity Frames

Ocean circulation models (e.g. FVCOM, NEMO) align numerical grids along regional bathymetric isobaths through Oblique Mercator or Lambert Conformal Conic projections. Eastward and northward velocities $(u_{\text{east}}, v_{\text{north}})$ must be rotated into along-shore and cross-shore grid velocities:
$\begin{pmatrix} u_{\text{grid}} \\ v_{\text{grid}} \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} u_{\text{east}} \\ v_{\text{north}} \end{pmatrix}$
"""

# ╔═╡ 00000000-0000-0000-0000-000000000008
begin
    # Station coordinates on the Scotian Shelf
    stn_coords = [(-64.0, 43.5), (-63.0, 44.0), (-62.0, 44.5)]
    theta_shelf = -0.45 # Rotation angle in radians (~25.8° clockwise)

    # Simulated geostrophic surface currents (eastward, northward in m/s)
    u_east = [0.25, 0.18, 0.30]
    v_north = [-0.10, -0.05, -0.15]

    # Rotate velocity vectors into along-shelf / cross-shelf frame
    u_grid = zeros(Float64, length(stn_coords))
    v_grid = zeros(Float64, length(stn_coords))
    for i in eachindex(stn_coords)
        u_grid[i] = cos(theta_shelf) * u_east[i] - sin(theta_shelf) * v_north[i]
        v_grid[i] = sin(theta_shelf) * u_east[i] + cos(theta_shelf) * v_north[i]
    end

    # Visualizing geographic vs rotated velocity vectors
    p_geo = quiver(
        [c[1] for c in stn_coords], [c[2] for c in stn_coords],
        quiver = (u_east .* 1.5, v_north .* 1.5),
        title = "Geographic Frame (East / North)",
        xlabel = "Longitude (°E)", ylabel = "Latitude (°N)",
        color = :blue, lw = 2.0
    )
    p_rot = quiver(
        [c[1] for c in stn_coords], [c[2] for c in stn_coords],
        quiver = (u_grid .* 1.5, v_grid .* 1.5),
        title = "Rotated Frame (Along / Cross Shelf)",
        xlabel = "Longitude (°E)", ylabel = "Latitude (°N)",
        color = :darkgreen, lw = 2.0
    )
    plot(p_geo, p_rot, layout = (1, 2), size = (850, 360))
end

# ╔═╡ 00000000-0000-0000-0000-000000000009
md"""
## 4. Coastal Boundaries, Marine Masking, and Wave Speeds

`Tethys` bundles validated coastal contours (`SCOTIAN_SHELF_COASTLINE`) and provides ray-casting point-in-polygon verification (`is_marine_water`). Restricting maximum water depth $H_{\max}$ constrains the barotropic gravity wave speed:
$$c_{\text{grav}} = \sqrt{g H_{\max}} \le 121.3\,\text{m/s}$$
ensuring numerical stability in free-surface split-explicit time stepping.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000010
begin
    # Synthesize station points spanning coastal and offshore waters
    rng_mask = MersenneTwister(101)
    pts_lon = rand(rng_mask, 150) .* 4.0 .- 66.0
    pts_lat = rand(rng_mask, 150) .* 3.0 .+ 43.5

    # Evaluate marine water status against bundled coastline rings
    marine_status = [
        is_marine_water(pts_lon[i], pts_lat[i]; coastline = SCOTIAN_SHELF_COASTLINE)
        for i in 1:150
    ]

    p_coast = plot(
        title = "Coastline Delineation & Marine Water Classification",
        xlabel = "Longitude (°E)",
        ylabel = "Latitude (°N)"
    )

    # Draw mainland and island polygons
    for ring in SCOTIAN_SHELF_COASTLINE
        plot!(p_coast, ring.lons, ring.lats, color = :gray30, lw = 1.2, fill = (0, 0.25, :gray70))
    end

    # Plot marine (blue) vs terrestrial/inland (orange) points
    scatter!(p_coast, pts_lon[marine_status], pts_lat[marine_status], color = :dodgerblue, ms = 4, label = "Marine Water")
    scatter!(p_coast, pts_lon[.!marine_status], pts_lat[.!marine_status], color = :darkorange, ms = 5, marker = :square, label = "Terrestrial Land")
    p_coast
end

# ╔═╡ 00000000-0000-0000-0000-000000000011
md"""
## 5. Spatial Discretization and Tessellation Algorithms

Comparing three tessellation strategies in `Tethys.Partitioning`:
- **`:hexagonal`**: Honeycomb cells maximizing area-to-perimeter ratios with uniform 6-neighbor topology.
- **`:cvt`**: Centroidal Voronoi Tessellation via Lloyd's relaxation, minimizing geometric quantization energy.
- **`:kvt`**: K-Means density-balanced Voronoi, concentrating units in high-density survey areas.

Visualized directly using canonical `TethysPlots.plot_partition`.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000012
begin
    rng_tess = MersenneTwister(2024)
    N_obs = 350
    # Clustering observations to demonstrate density-adaptive behavior
    c1_x = randn(rng_tess, 200) .* 0.8 .- 63.5; c1_y = randn(rng_tess, 200) .* 0.6 .+ 44.2
    c2_x = randn(rng_tess, 150) .* 1.2 .- 61.0; c2_y = randn(rng_tess, 150) .* 0.8 .+ 45.0
    obs_df = DataFrame(lon = vcat(c1_x, c2_x), lat = vcat(c1_y, c2_y))

    au_hex = assign_spatial_units(obs_df; x = :lon, y = :lat, area_method = :hexagonal, target_units = 14, exact_units = true)
    au_cvt = assign_spatial_units(obs_df; x = :lon, y = :lat, area_method = :cvt, target_units = 14)
    au_kvt = assign_spatial_units(obs_df; x = :lon, y = :lat, area_method = :kvt, target_units = 14)

    # Render tessellations using canonical TethysPlots.plot_partition
    p_hex = plot_partition(au_hex; data = obs_df, x = :lon, y = :lat, title = "Hexagonal (:hexagonal)")
    p_cvt = plot_partition(au_cvt; data = obs_df, x = :lon, y = :lat, title = "Centroidal Voronoi (:cvt)")
    p_kvt = plot_partition(au_kvt; data = obs_df, x = :lon, y = :lat, title = "K-Means Density (:kvt)")

    plot(p_hex, p_cvt, p_kvt, layout = (1, 3), size = (1000, 340))
end

# ╔═╡ 00000000-0000-0000-0000-000000000013
md"""
## 6. Spatial Graph Topology, Island Bridging, and BYM2 Scaling

Intrinsic Autoregressive (ICAR) and Besag-York-Mollié (BYM2) models require a connected spatial graph $\mathcal{G}$ with a single rank deficiency. `ensure_connected!` connects disjoint marine islands via KD-tree minimum Euclidean links, and `scaling_factor_bym2` computes the geometric mean scaling factor:
$$\tau = \exp\left( -\frac{1}{S-1} \sum_{i=1}^{S-1} \ln \lambda_i \right)$$

Rendered using canonical `TethysPlots.spatial_graph_plot`.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000014
begin
    centroids = au_hex.centroids
    W_sparse = spatial_weights_matrix(centroids; method = :knn, k = 4)

    # Compute BYM2 spectral precision scale factor
    tau_scale = scaling_factor_bym2(W_sparse)

    # Visualize graph connectivity network via canonical TethysPlots.spatial_graph_plot
    p_graph = spatial_graph_plot(
        au_hex.polygons, W_sparse;
        centroids = centroids,
        title = "Spatial Adjacency Network (τ = $(round(tau_scale, digits=4)))",
        edge_color = :steelblue,
        edge_lw = 1.3,
        poly_linecolor = :gray80
    )
end

# ╔═╡ 00000000-0000-0000-0000-000000000015
md"""
## 7. Spatiotemporal Synchronization

`assign_spatiotemporal_units` maps multi-year observational data into discrete space-time cells using the indexing formula:
$$st = (t - 1) \cdot S + s$$
where $t \in \{1, \dots, T\}$ denotes the temporal slice and $s \in \{1, \dots, S\}$ denotes the spatial unit.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000016
begin
    # Synthesize multi-year survey records
    years = rand(MersenneTwister(999), 2020:2024, nrow(obs_df))
    obs_df.year = years

    st_results = assign_spatiotemporal_units(
        obs_df;
        space_x = :lon,
        space_y = :lat,
        time_var = :year,
        area_method = :hexagonal,
        target_units = 12,
        exact_units = true,
        time_method = "unique"
    )

    S_count = length(st_results.au_spatial.centroids)
    T_count = st_results.au_temporal.N_cat

    # Observation count matrix across space-time units
    st_matrix = zeros(Int, S_count, T_count)
    for i in 1:nrow(obs_df)
        s = st_results.s_idx[i]
        t = st_results.t_idx[i]
        st_matrix[s, t] += 1
    end

    # Space-time sampling intensity heatmap
    p_st_heat = heatmap(
        2020:2024, 1:S_count, st_matrix,
        title = "Observation Counts by Space-Time Unit",
        xlabel = "Year (t)",
        ylabel = "Spatial Unit Index (s)",
        colorbar = true
    )

    # Spatial choropleth of marginal observation totals via TethysPlots.choropleth
    spatial_totals = vec(sum(st_matrix, dims = 2))
    p_st_choro = choropleth(
        st_results.au_spatial.polygons, spatial_totals;
        title = "Marginal Spatial Intensity (Total Obs)",
        cmap = :plasma,
        colorbar_label = "Observations",
        show_colorbar = true
    )

    plot(p_st_heat, p_st_choro, layout = (1, 2), size = (900, 380))
end

# ╔═╡ 00000000-0000-0000-0000-000000000017
md"""
## 8. Cross-Mesh Topological Transfer Operators (Resharding)

In multi-tiered oceanographic models, different data products exist at different spatial scales. `compute_network_transfer_matrix` constructs a linear transfer operator $P \in \mathbb{R}^{N_{\text{dest}} \times N_{\text{src}}}$ using geometric area-weighted intersection:
$$P_{j, i} = \frac{\text{Area}(B_j \cap A_i)}{\text{Area}(B_j)}, \quad \sum_{i=1}^{N_{\text{src}}} P_{j, i} = 1$$
mapping fields smoothly between coarse and fine meshes ($\mathbf{u}_{\text{dest}} = P \mathbf{u}_{\text{src}}$).
"""

# ╔═╡ 00000000-0000-0000-0000-000000000018
begin
    # Construct coarse mesh (10 units) and fine assessment mesh (30 units)
    au_coarse = assign_spatial_units(obs_df.lon, obs_df.lat; area_method = :cvt, target_units = 10)
    au_fine = assign_spatial_units(obs_df.lon, obs_df.lat; area_method = :hexagonal, target_units = 30)

    # Compute transfer operator matrix P [N_dest x N_src]
    P_transfer = compute_network_transfer_matrix(au_coarse, au_fine; method = :area_weighted)

    # Simulate latent environmental field on coarse mesh
    u_coarse = [Float64(sin(c[1] * 0.5) + cos(c[2] * 0.5)) for c in au_coarse.centroids]

    # Map field onto fine mesh via linear transfer operator
    u_fine = P_transfer * u_coarse

    # Visualizing coarse vs fine resharded field using canonical TethysPlots.choropleth
    p_c = choropleth(
        au_coarse.polygons, u_coarse;
        title = "Coarse Mesh (10 units)",
        cmap = :viridis,
        show_colorbar = false
    )

    p_f = choropleth(
        au_fine.polygons, u_fine;
        title = "Resharded Fine Mesh (30 units)",
        cmap = :viridis,
        show_colorbar = true,
        colorbar_label = "Field Value"
    )

    plot(p_c, p_f, layout = (1, 2), size = (850, 380))
end

# ╔═╡ 00000000-0000-0000-0000-000000000019
md"""
## 9. Hierarchical Analytical Lakehouse Persistence

`GeoStorage` exposes an analytical storage API for storing simulation arrays into Zarr groups. The cell below creates a lakehouse group in a temporary directory, persists multidimensional variables, and verifies round-trip fidelity.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000020
begin
    mktempdir() do tmp_dir
        store_file = joinpath(tmp_dir, "lakehouse_demo.zarr")
        gs = open_geostorage(store_file; backend = :zarr, create = true)

        # Create simulation run group
        create_storage_group(gs, "simulation_2024")

        # Persist spatial variable slice
        test_payload = [Float32(i * 1.5 + j) for i in 1:20, j in 1:15]
        write_storage_variable!(gs, "simulation_2024", "hydrography_temp", test_payload)

        # Read back stored variable
        read_back = read_storage_variable(gs, "simulation_2024", "hydrography_temp")
        close_geostorage(gs)

        (
            storage_format = "Zarr (DirectoryStore)",
            group_name = "simulation_2024",
            variable = "hydrography_temp",
            payload_shape = size(test_payload),
            verified_match = (read_back == test_payload)
        )
    end
end

# ╔═╡ 00000000-0000-0000-0000-000000000021
md"""
## 10. Interactive GIS Mapping with Leaflet.js

`TethysPlots` generates standalone, zero-dependency interactive HTML map dashboards powered by Leaflet.js. Polygons are rendered with tile overlays, interactive tooltips, and customizable palettes.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000022
begin
    # Construct interactive Leaflet choropleth for hexagonal units
    m_leaf = leaflet_choropleth(
        au_hex.polygons, [c[1] for c in au_hex.centroids];
        title = "Scotian Shelf Hexagonal Lattice",
        colorbar_label = "Longitude Centroid (°E)",
        cmap = :viridis,
        line_weight = 1.0,
        fill_opacity = 0.65
    )
    "Interactive LeafletMap created with $(length(au_hex.polygons)) polygonal cells."
end

# ╔═╡ 00000000-0000-0000-0000-000000000023
md"""
## 11. Hydrodynamic Velocity Fields and Oceanographic Sections

Physical oceanography workflows require visualization of velocity vector fields $(u, v)$ and vertical hydrographic stratification across the water column. `TethysPlots` provides dedicated primitives:
- `plot_advection_arrows`: Renders vector arrows scaled by hydrodynamic current speed.
- `plot_hydrodynamic_section`: Renders vertical transects ($z \times x$) with seafloor bedrock bathymetry masking.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000024
begin
    # 1. Surface advection velocity field
    cents = au_hex.centroids
    vx = [0.15 * cos(c[2] * 2π / 5.0) for c in cents]
    vy = [0.10 * sin(c[1] * 2π / 5.0) for c in cents]
    p_arrows = plot_advection_arrows(
        au_hex.polygons, vx, vy;
        centroids = cents,
        title = "Surface Advection Velocity Field",
        arrow_color = :navy,
        scale = 0.8
    )

    # 2. Vertical hydrographic temperature section
    stn_lons = collect(range(-65.0, -61.0, length = 5))
    depth_layers = collect(range(0.0, 300.0, length = 10))
    temp_sec = [Float64(15.0 * exp(-z / 80.0) + 2.0 * sin(x)) for x in stn_lons, z in depth_layers]
    bathy = [120.0, 180.0, 250.0, 220.0, 150.0]
    hydro_obj = (temperature = temp_sec, seafloor = bathy)
    au_stns = (centroids = [[x, 44.0] for x in stn_lons],)

    p_sec = plot_hydrodynamic_section(
        hydro_obj, au_stns;
        var = :temperature,
        along = :lon,
        z_coords = depth_layers,
        cmap = :thermal,
        title = "Hydrodynamic Temperature Section across Scotian Shelf"
    )

    plot(p_arrows, p_sec, layout = (1, 2), size = (1000, 420))
end
