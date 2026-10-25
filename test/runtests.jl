# ==============================================================================
#  Tethys Test Suite
# ==============================================================================

using Test
using CoordRefSystems
using Dates
using LinearAlgebra
using SparseArrays
using Statistics
using Tethys

const ALL_SEGMENTS = [:types, :config, :cli, :crs, :storage, :api, :partitioning]

function parse_test_segments(args::Vector{String})
    isempty(args) && return ALL_SEGMENTS
    segments = Symbol[]
    for a in args
        s = Symbol(lowercase(strip(a)))
        if s in (:all, :full)
            return ALL_SEGMENTS
        elseif s in ALL_SEGMENTS
            push!(segments, s)
        end
    end
    return isempty(segments) ? ALL_SEGMENTS : segments
end

const ACTIVE_SEGMENTS = parse_test_segments(ARGS)
println("Running Tethys test segments: ", join(ACTIVE_SEGMENTS, ", "))

@testset "Tethys Comprehensive Test Suite" begin

    if :types in ACTIVE_SEGMENTS
        @testset "Types: Dimension, Array, and Dataset" begin
            d_lon = Dimension(name = :lon, size = 10, coords = range(-70.0, -60.0, length = 10))
            d_lat = Dimension(name = :lat, size = 8, coords = range(40.0, 47.0, length = 8))
            d_time = Dimension(name = :time, size = 5, dim_type = DIM_TEMPORAL)

            @test d_lon.name == :lon
            @test d_lon.size == 10
            @test d_time.dim_type == DIM_TEMPORAL

            crs = CoordinateSystem(crs = "EPSG:4326")
            @test !is_cartesian(crs)
            @test is_cartesian(CoordinateSystem())

            data = rand(10, 8)
            ga = GeoArray(data; dims = [d_lon, d_lat], crs = crs)
            @test size(ga) == (10, 8)
            @test ndims(ga) == 2
            @test ga[1, 1] == data[1, 1]

            ds = GeoDataset(
                Dict("temperature" => ga),
                Dict(:lon => GeoArray(collect(d_lon.coords); dims = [d_lon]),
                     :lat => GeoArray(collect(d_lat.coords); dims = [d_lat])),
                Dict(:lon => d_lon, :lat => d_lat),
                crs,
                Dict{String, Any}("title" => "Test Dataset"),
                nothing,
                "memory://test"
            )

            @test haskey(ds, "temperature")
            @test ds["temperature"] isa GeoArray
            @test ds.temperature isa GeoArray
            @test haskey(ds.coords, :lon)
            @test ds.dims[:lon].size == 10
        end
    end

    if :config in ACTIVE_SEGMENTS
        @testset "Config: TOML and Path Resolution" begin
            mktempdir() do tmp
                cfg_path = joinpath(tmp, "test_config.toml")
                test_dict = Dict(
                    "project" => Dict("name" => "TethysRun", "threads" => 4),
                    "storage" => Dict("store_root" => "data/lakehouse", "verify" => true)
                )
                save_config(cfg_path, test_dict)
                @test isfile(cfg_path)

                loaded = load_config(cfg_path)
                @test loaded["project"]["name"] == "TethysRun"
                @test get_config_val(loaded, "project", "threads", 1) == 4
                @test get_config_val(loaded, "project", "missing_key", 99) == 99
                @test get_config_val(loaded, "nonexistent_section", "k", "def") == "def"

                resolved = resolve_config_path(loaded, "relative/path/grid.zarr")
                @test isabspath(resolved)
                @test endswith(resolved, joinpath("relative", "path", "grid.zarr"))

                @test validate_config(loaded, ["project", "storage"])
                @test_throws ErrorException validate_config(loaded, ["required_missing"])
            end
        end
    end

    if :cli in ACTIVE_SEGMENTS
        @testset "CLI: Argument Parsing and Formatting" begin
            args = ["--config=test.toml", "--output-dir=results", "--overwrite", "--grid"]
            opts = parse_cli_args(args; default_config = "default.toml")

            @test opts["config"] == "test.toml"
            @test opts["output_dir"] == "results"
            @test opts["overwrite"] == true
            @test opts["grid"] == true
            @test :grid in opts["actions"]

            buf = IOBuffer()
            display_cli_summary(buf, "TethysTest", "Geospatial Execution Engine",
                                Dict("grid" => "Build spatial mesh"))
            summary_str = String(take!(buf))
            @test occursin("TethysTest", summary_str)
            @test occursin("--config=FILE", summary_str)
            @test occursin("--grid", summary_str)
        end
    end

    if :crs in ACTIVE_SEGMENTS
        @testset "CRS: Resolution, Projections, and Vector Rotation" begin
            crs_wgs = resolve_crs("EPSG:4326")
            @test crs_wgs !== nothing
            @test crs_wgs <: CoordRefSystems.CRS

            # UTM zone 20N (EPSG:32620)
            proj_utm = build_projection("EPSG:32620")
            @test is_planar(proj_utm)

            # Halifax coordinates: -63.57° E, 44.65° N
            pt_xy = project_forward(proj_utm, -63.57, 44.65)
            @test pt_xy[1] > 400000.0
            @test pt_xy[2] > 4900000.0

            pt_geo = project_inverse(proj_utm, pt_xy[1], pt_xy[2])
            @test isapprox(pt_geo[1], -63.57; atol = 1e-4)
            @test isapprox(pt_geo[2], 44.65; atol = 1e-4)

            # Conformal Lambert Conic
            proj_lcc = build_projection(:lambert; lon_0 = -60.0, lat_0 = 45.0,
                                        lat_1 = 43.0, lat_2 = 47.0)
            @test is_planar(proj_lcc)
            lcc_xy = project_forward(proj_lcc, -60.0, 45.0)
            @test isapprox(lcc_xy[1], 0.0; atol = 1.0)
            @test isapprox(lcc_xy[2], 0.0; atol = 1.0)

            # Rotated Frame (Shelf alignment angle)
            rot = RotatedFrame(lon_center = -60.0, lat_center = 45.0, angle_deg = -30.0)
            @test rot.angle_deg == -30.0

            # Vector rotation round-trip
            u_geo, v_geo = 1.5, -0.8
            u_rot, v_rot = rotate_vector(u_geo, v_geo, 45.0)
            u_back, v_back = unrotate_vector(u_rot, v_rot, 45.0)
            @test isapprox(u_back, u_geo; atol = 1e-10)
            @test isapprox(v_back, v_geo; atol = 1e-10)
            @test isapprox(hypot(u_rot, v_rot), hypot(u_geo, v_geo); atol = 1e-10)
        end
    end

    if :storage in ACTIVE_SEGMENTS
        @testset "Storage: Zarr, Parquet, and GeoStorage Lakehouse" begin
            mktempdir() do tmp
                # Zarr round-trip
                zarr_path = joinpath(tmp, "test_dataset.zarr")
                d_x = Dimension(name = :lon, size = 5, coords = collect(1.0:5.0))
                d_y = Dimension(name = :lat, size = 4, coords = collect(10.0:13.0))
                arr = rand(5, 4)
                ga = GeoArray(arr; dims = [d_x, d_y])
                ds = GeoDataset(
                    Dict("var1" => ga),
                    Dict(:lon => GeoArray(collect(1.0:5.0); dims = [d_x]),
                         :lat => GeoArray(collect(10.0:13.0); dims = [d_y])),
                    Dict(:lon => d_x, :lat => d_y)
                )

                save_dataset(zarr_path, ds; backend = :zarr, overwrite = true)
                @test isdir(zarr_path)

                loaded_ds = load_dataset(zarr_path; backend = :zarr)
                @test haskey(loaded_ds, "var1")
                @test size(loaded_ds["var1"]) == (5, 4)
                @test isapprox(loaded_ds["var1"][:, :], arr; atol = 1e-8)

                # GeoStorage abstraction
                gs_path = joinpath(tmp, "archive.zarr")
                gs = open_geostorage(gs_path; backend = :zarr, create = true)
                create_storage_group(gs, "simulation_01")
                @test has_storage_group(gs, "simulation_01")
                write_storage_variable!(gs, "simulation_01", "pressure", [101.3, 101.5, 102.0])
                read_back = read_storage_variable(gs, "simulation_01", "pressure")
                @test read_back == [101.3, 101.5, 102.0]
                close_geostorage(gs)
            end
        end
    end

    if :api in ACTIVE_SEGMENTS
        @testset "API: Dataset Slicing, Sampling, and Operations" begin
            lons = collect(range(-70.0, -60.0, length = 11))
            lats = collect(range(40.0, 50.0, length = 11))
            d_lon = Dimension(name = :lon, size = 11, coords = lons)
            d_lat = Dimension(name = :lat, size = 11, coords = lats)
            field = [Float64(i + 2 * j) for i in 1:11, j in 1:11]
            ga = GeoArray(field; dims = [d_lon, d_lat])

            ds = GeoDataset(
                Dict("elevation" => ga),
                Dict(:lon => GeoArray(lons; dims = [d_lon]),
                     :lat => GeoArray(lats; dims = [d_lat])),
                Dict(:lon => d_lon, :lat => d_lat)
            )

            # Spatial slicing
            sliced = slice_dataset(ds; lon = (-68.0, -62.0), lat = (42.0, 48.0))
            @test haskey(sliced, "elevation")
            @test size(sliced["elevation"], 1) < 11
            @test size(sliced["elevation"], 2) < 11

            # Point query / sample
            vals = sample_dataset(ds, ["elevation"]; lon = -65.0, lat = 45.0)
            @test haskey(vals, "elevation")
            @test vals["elevation"] isa Real
        end
    end

    if :partitioning in ACTIVE_SEGMENTS
        @testset "Partitioning: HexMesh and Spatial Graphs" begin
            # Hexagonal mesh generation
            bbox = (-65.0, -60.0, 42.0, 45.0)
            mesh = build_hex_mesh(bbox; cell_size_km = 50.0)

            @test haskey(mesh, :centroids_lonlat)
            @test haskey(mesh, :polygons_lonlat)
            @test length(mesh.centroids_lonlat) > 0
            @test length(mesh.centroids_lonlat) == length(mesh.polygons_lonlat)

            # Coordinate projection helpers
            pt_xy = lonlat_to_xy_km(-64.0, 43.5; crs = "EPSG:32620")
            @test pt_xy[1] isa Real
            @test pt_xy[2] isa Real
            pt_lonlat = xy_km_to_lonlat(pt_xy[1], pt_xy[2]; crs = "EPSG:32620")
            @test isapprox(pt_lonlat[1], -64.0; atol = 1e-4)
            @test isapprox(pt_lonlat[2], 43.5; atol = 1e-4)

            # Spatial weights matrix (k-nearest neighbors)
            W = spatial_weights_matrix(mesh.centroids_lonlat; method = :knn, k = 4)
            @test size(W, 1) == length(mesh.centroids_lonlat)
            @test size(W, 2) == length(mesh.centroids_lonlat)
            @test issparse(W)
        end
    end

end
