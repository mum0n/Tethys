"""
High-level API: joining datasets.
"""

export join_datasets, geojoin, geomerged

"""
    join_datasets(datasets::Vector{GeoDataset}; on::Vector{Symbol}, how::Symbol=:inner) -> GeoDataset

Join multiple datasets along shared dimensions.

# Arguments
- `datasets`: Datasets to join
- `on`: Dimension names to join on (e.g., `[:time]`, `[:lon, :lat]`)
- `how`: Join type - `:inner` (intersection), `:outer` (union), `:left`, `:right`

# Returns
Joined `GeoDataset`.

# Examples
```julia
ds_joined = join_datasets([ds1, ds2], on=[:time])
ds_joined = join_datasets([ds1, ds2], on=[:lon, :lat], how=:outer)
```
"""
function join_datasets(datasets::Vector{GeoDataset}; on::Vector{Symbol}, how::Symbol=:inner)
    return Operations.join_datasets(datasets; on=on, how=how)
end

const geojoin = join_datasets

# Convenience: merge variables from multiple datasets with same grid
"""
    geomerged(datasets::Vector{GeoDataset}) -> GeoDataset

Merge variables from datasets that share the same coordinate grid.
"""
function geomerged(datasets::Vector{GeoDataset})
    length(datasets) >= 2 || error("Need at least 2 datasets to merge")
    
    # Check they have compatible coordinates
    ref = datasets[1]
    for ds in datasets[2:end]
        for (dim, d) in ref.dims
            haskey(ds.dims, dim) || error("Dataset missing dimension: $dim")
            ds.dims[dim].size == d.size || error("Dimension $dim size mismatch")
        end
    end
    
    # Merge variables (rename collisions)
    vars = Dict{String, GeoArray}()
    for (i, ds) in enumerate(datasets)
        for (name, ga) in ds.variables
            new_name = length(datasets) == 1 ? name : "$(name)_ds$i"
            vars[new_name] = ga
        end
    end
    
    return GeoDataset(vars, ref.coords, ref.dims, ref.crs, ref.attrs, ref.backend, join([ds.source for ds in datasets], "+"))
end