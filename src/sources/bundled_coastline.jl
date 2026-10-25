"""
    SCOTIAN_SHELF_COASTLINE

Coastline rings for the Scotian Shelf, the Gulf of St. Lawrence and the adjacent northwestern
Atlantic, as a vector of `(name, code, lons, lats)` NamedTuples.

The rings are closed polygonal perimeters -- a coastline is a set of open LineStrings in Natural
Earth, and `point_in_polygon` needs a ring. They are shipped rather than fetched so an offline
run can still reject land; a run that needs a real shoreline fetches one through
`fetch_natural_earth_coastline` and uses that instead.

These rings are the region this project models, so the name says so. The general land test in
`Tethys` takes whatever rings it is given and applies no default: deciding that a Scotian
Shelf run should fall back to these rings is the regional application's decision, and the
application layers (`TethysFlow`, `TethysDrift`) make it
explicitly by passing them.
"""
const SCOTIAN_SHELF_COASTLINE = [
    # 1. Nova Scotia Mainland (clockwise closed perimeter)
    (
        name = "Nova Scotia Mainland",
        code = :nova_scotia_mainland,
        lons = [
            -64.25, -64.95, -64.49, -64.36, -65.75, -66.35, -66.05, -66.15,
            -65.98, -65.62, -65.32, -64.60, -64.30, -63.92, -63.55, -63.45,
            -63.00, -62.50, -61.98, -61.40, -60.98, -61.50, -61.40, -61.90,
            -62.25, -62.70, -63.13, -63.30, -63.67, -64.21, -64.25
        ],
        lats = [
            45.75,  45.33,  45.33,  45.09,  44.65,  44.27,  44.30,  43.80,
            43.70,  43.47,  43.70,  44.10,  44.40,  44.49,  44.46,  44.60,
            44.72,  44.88,  45.00,  45.18,  45.33,  45.38,  45.60,  45.87,
            45.65,  45.68,  45.79,  45.75,  45.85,  45.83,  45.75
        ]
    ),
    # 2. Cape Breton Island (clockwise from Strait of Canso)
    (
        name = "Cape Breton Island",
        code = :cape_breton,
        lons = [
            -61.40, -61.53, -61.50, -61.12, -61.00, -60.80, -60.50, -60.42,
            -60.45, -60.40, -60.32, -60.20, -59.70, -59.80, -60.15, -60.50,
            -60.85, -61.00, -61.20, -61.40
        ],
        lats = [
            45.65,  45.98,  46.07,  46.43,  46.63,  46.83,  47.05,  47.03,
            46.90,  46.65,  46.33,  46.20,  46.03,  45.92,  45.85,  45.75,
            45.65,  45.55,  45.55,  45.65
        ]
    ),
    # 3. Prince Edward Island (clockwise from West Point)
    (
        name = "Prince Edward Island",
        code = :prince_edward_island,
        lons = [
            -64.40, -63.99, -63.00, -61.97, -62.25, -62.46, -62.78, -63.49,
            -63.70, -63.85, -64.40
        ],
        lats = [
            46.62,  47.05,  46.50,  46.45,  46.35,  46.00,  45.95,  46.20,
            46.25,  46.38,  46.62
        ]
    ),
    # 4. New Brunswick, Quebec & Maine Continental Mainland
    # Extended west to -71°E and north to 48.5°N to cover the expanded domain.
    (
        name = "NB-Quebec-Maine Continental Mainland",
        code = :new_brunswick_mainland,
        lons = [
            -64.21, -64.35, -64.79, -65.53, -66.00, -66.47, -67.05, -67.00,
            -67.45, -67.80, -68.50, -69.10, -69.80, -70.20, -70.60, -71.00,
            -71.00, -70.50, -70.00, -69.50, -69.00, -68.50, -68.00, -67.60,
            -67.00, -66.50, -66.00, -65.50, -65.00, -64.85, -64.50, -64.10,
            -64.21
        ],
        lats = [
            45.83,  45.75,  45.60,  45.35,  45.20,  45.06,  45.08,  44.88,
            44.65,  44.45,  44.30,  44.10,  43.80,  43.58,  43.50,  43.00,
            48.50,  48.50,  48.50,  48.50,  48.50,  48.50,  48.50,  48.50,
            48.50,  48.50,  48.50,  47.90,  47.10,  46.68,  46.25,  46.00,
            45.83
        ]
    ),
    # 5. Southern & Western Newfoundland
    # Expanded to cover both Avalon Peninsula and western Newfoundland coasts
    # visible in the domain north of 46.5°N and east of -60°W.
    (
        name = "Southern-Western Newfoundland",
        code = :newfoundland_south,
        lons = [
            -59.13, -58.70, -57.60, -56.00, -55.20, -54.20, -53.40, -53.05,
            -52.70, -53.00, -53.50, -54.10, -54.80, -55.60, -56.40, -57.30,
            -58.00, -58.40, -58.80, -59.13
        ],
        lats = [
            47.57,  47.61,  47.61,  47.55,  47.10,  46.80,  46.70,  46.65,
            47.55,  48.00,  48.30,  48.50,  48.50,  48.50,  48.50,  48.50,
            48.50,  48.30,  48.00,  47.57
        ]
    ),
    # 6. Newfoundland Avalon Peninsula (eastern Newfoundland protruding into domain)
    (
        name = "Newfoundland Avalon Peninsula",
        code = :newfoundland_avalon,
        lons = [
            -52.70, -52.70, -53.30, -53.60, -53.20, -52.90, -52.80, -52.70
        ],
        lats = [
            47.55,  48.00,  48.00,  47.50,  46.65,  46.80,  47.10,  47.55
        ]
    ),
    # 7. Anticosti Island (Gulf of St. Lawrence)
    (
        name = "Anticosti Island",
        code = :anticosti,
        lons = [
            -63.60, -64.20, -64.30, -63.60, -62.80, -62.00, -61.60, -61.80,
            -62.30, -62.80, -63.60
        ],
        lats = [
            49.90,  49.85,  49.45,  49.10,  49.20,  49.35,  49.60,  49.90,
            49.95,  50.00,  49.90
        ]
    ),
    # 8. Magdalen Islands (Îles-de-la-Madeleine) — clustered archipelago
    (
        name = "Magdalen Islands",
        code = :magdalen_islands,
        lons = [-62.00, -61.60, -61.50, -61.80, -62.10, -62.40, -62.00],
        lats = [ 47.30,  47.20,  47.50,  47.65,  47.55,  47.35,  47.30]
    ),
    # 9. Sable Island (closed perimeter)
    (
        name = "Sable Island",
        code = :sable_island,
        lons = [-60.15, -59.90, -59.70, -59.90, -60.15],
        lats = [43.93, 43.95, 43.96, 43.91, 43.93]
    )
]
