export aggregate_dataset, geoaggregate

"""
    aggregate_dataset(ds::GeoDataset; dim::Symbol, func::Function) -> GeoDataset

Reduce every variable along `dim` and return a dataset with that dimension removed.

`func` must accept the array-positional form `func(data; dims=i)`, so `mean`, `sum`,
`maximum`, and `extrema` all work. A variable that does not span `dim` is returned
unchanged rather than dropped, and the reduction is recorded in `attrs["reduced"]`.
"""
aggregate_dataset(ds::GeoDataset; dim::Symbol, func::Function) =
    aggregate(ds; dim = dim, func = func)

const geoaggregate = aggregate_dataset
