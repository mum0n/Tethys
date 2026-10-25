# ==============================================================================
# columns.jl -- (GeoPartitioning)
#
# Spatial/temporal column-name vocabulary and automatic column resolvers,
# extracted verbatim from TethysBayes src/definitions.jl (lines 534-741), where they
# were the only coupling partitioning.jl had to the rest of the package. The
# remaining `_detect_*` resolvers (response, spatial-unit, seasonal, group)
# stay in TethysBayes: they describe modelling vocabulary, not domain decomposition.
# ==============================================================================

# -----------------------------------------------------------------------------
# Coordinate & Variable Resolvers
# -----------------------------------------------------------------------------

"""
Standard pairs of spatial 2D coordinate variable names, ordered by preference.
Used for automatic spatial coordinate detection when users provide custom column names.
"""
const STANDARD_SPATIAL_COORDINATE_PAIRS = [
    (:s_x, :s_y),
    (:plon, :plat),
    (:lon, :lat),
    (:longitude, :latitude),
    (:easting, :northing),
    (:x, :y),
    (:coord_x, :coord_y),
    (:coords_x, :coords_y),
    (:spatial_x, :spatial_y),
    (:s1, :s2)
]

"""
Standard candidate column names for temporal indexing, ordered by preference.
"""
const STANDARD_TEMPORAL_CANDIDATES = [
    :t_idx,
    :year,
    :time,
    :timestamp,
    :date,
    :datetime,
    :t,
    :day,
    :week,
    :month,
    :hour,
    :step
]

"""
    _detect_xy_columns(df; x=nothing, y=nothing)::Tuple{Symbol, Symbol}

Automatically identifies or validates 2D spatial coordinate columns in a tabular dataset
or NamedTuple `df`.

# Process & Mathematical Context
Spatial models and continuous smooths operate on spatial point coordinates
\$\\mathbf{s}_i = (x_i, y_i) \\in \\mathbb{R}^2\$. Users commonly label their coordinate
axes with domain-specific conventions (e.g. `plon`/`plat` for oceanographic bathymetry,
`easting`/`northing` for projected UTM cartesian grids, or `lon`/`lat` for geodetic
data). This function systematically resolves the coordinate column names:
1. If both `x` and `y` are explicitly provided, it validates that they exist in `df`.
2. If either `x` or `y` is unspecified, it matches against known coordinate pairs
   defined in `STANDARD_SPATIAL_COORDINATE_PAIRS`.
3. If no recognized coordinate columns are present, it raises a descriptive `ArgumentError`.

# Inputs
- `df`: Tabular dataset (`DataFrame`, `NamedTuple`, or table-like object).
- `x`: Optional explicit name for the horizontal/x coordinate column.
- `y`: Optional explicit name for the vertical/y coordinate column.

# Outputs
- `Tuple{Symbol, Symbol}`: Resolved `(col_x, col_y)` symbols.
"""
function _detect_xy_columns(
    df; x=nothing, y=nothing, allow_nothing::Bool=false
)::Union{Tuple{Symbol, Symbol}, Nothing}
    # 1. Explicitly provided both x and y
    if !isnothing(x) && !isnothing(y)
        sx, sy = Symbol(x), Symbol(y)
        if !hasproperty(df, sx)
            allow_nothing && return nothing
            error(
                "Specified coordinate column x=:$sx not found in data: " *
                "$(propertynames(df))."
            )
        end
        if !hasproperty(df, sy)
            allow_nothing && return nothing
            error(
                "Specified coordinate column y=:$sy not found in data: " *
                "$(propertynames(df))."
            )
        end
        return (sx, sy)
    end

    # 2. If only one is specified, find matching partner in standard pairs
    if !isnothing(x) && isnothing(y)
        sx = Symbol(x)
        if !hasproperty(df, sx)
            allow_nothing && return nothing
            error(
                "Specified coordinate column x=:$sx not found in data: " *
                "$(propertynames(df))."
            )
        end
        for (cx, cy) in STANDARD_SPATIAL_COORDINATE_PAIRS
            if cx == sx && hasproperty(df, cy)
                return (sx, cy)
            elseif cy == sx && hasproperty(df, cx)
                return (cx, sx)
            end
        end
        for cy in [:s_y, :northing, :lat, :latitude, :y, :plat, :coords_y]
            if hasproperty(df, cy) && cy != sx
                return (sx, cy)
            end
        end
        allow_nothing && return nothing
        error(
            "Coordinate column x=:$sx found, but could not detect matching " *
            "y coordinate column."
        )
    end

    if isnothing(x) && !isnothing(y)
        sy = Symbol(y)
        if !hasproperty(df, sy)
            allow_nothing && return nothing
            error(
                "Specified coordinate column y=:$sy not found in data: " *
                "$(propertynames(df))."
            )
        end
        for (cx, cy) in STANDARD_SPATIAL_COORDINATE_PAIRS
            if cy == sy && hasproperty(df, cx)
                return (cx, sy)
            elseif cx == sy && hasproperty(df, cy)
                return (cy, sy)
            end
        end
        for cx in [:s_x, :plon, :lon, :longitude, :easting, :x, :coords_x]
            if hasproperty(df, cx) && cx != sy
                return (cx, sy)
            end
        end
        allow_nothing && return nothing
        error(
            "Coordinate column y=:$sy found, but could not detect matching " *
            "x coordinate column."
        )
    end

    # 3. Neither specified: iterate standard candidate pairs
    for (cx, cy) in STANDARD_SPATIAL_COORDINATE_PAIRS
        if hasproperty(df, cx) && hasproperty(df, cy)
            return (cx, cy)
        end
    end

    if allow_nothing
        return nothing
    end

    error(
        "Could not automatically detect spatial coordinate columns in dataset. " *
        "Available columns: $(propertynames(df)). " *
        "Please specify coordinate columns explicitly " *
        "(e.g., x=:lon, y=:lat or x=:plon, y=:plat)."
    )
end

"""
    _detect_time_column(df; time_var=nothing, allow_nothing=false)::Union{Symbol, Nothing}

Automatically identifies or validates the temporal index or date column in a dataset.

# Inputs
- `df`: Tabular dataset.
- `time_var`: Optional explicit name for the temporal column.
- `allow_nothing`: If true, returns `nothing` when not detected instead of error.

# Outputs
- `Union{Symbol, Nothing}`: Resolved temporal column name or nothing.
"""
function _detect_time_column(
    df; time_var=nothing, allow_nothing::Bool=false
)::Union{Symbol, Nothing}
    if !isnothing(time_var)
        st = Symbol(time_var)
        if !hasproperty(df, st)
            allow_nothing && return nothing
            error(
                "Specified temporal column time_var=:$st not found in data: " *
                "$(propertynames(df))."
            )
        end
        return st
    end

    for t_cand in STANDARD_TEMPORAL_CANDIDATES
        if hasproperty(df, t_cand)
            return t_cand
        end
    end

    if allow_nothing
        return nothing
    end

    error(
        "Could not automatically detect temporal column in dataset. " *
        "Available columns: $(propertynames(df)). " *
        "Please specify temporal column explicitly " *
        "(e.g., time_var=:year or time_var=:time)."
    )
end
