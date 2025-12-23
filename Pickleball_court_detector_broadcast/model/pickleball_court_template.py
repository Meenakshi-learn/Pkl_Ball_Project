import numpy as np

def pickleball_court_world():
    """
    Pickleball court (standard)
    Units: meters
    Origin: bottom-left
    """

    points = {
        "BL": (0.0, 0.0),
        "BR": (6.10, 0.0),
        "TR": (6.10, 13.41),
        "TL": (0.0, 13.41),

        "NVZ_bottom": (0.0, 2.13),
        "NVZ_top": (0.0, 11.28),

        "center_bottom": (3.05, 0.0),
        "center_top": (3.05, 13.41),
    }

    lines = [
        ("BL", "BR"),
        ("BR", "TR"),
        ("TR", "TL"),
        ("TL", "BL"),
        ("NVZ_bottom", "NVZ_top"),
        ("center_bottom", "center_top"),
    ]

    return points, lines