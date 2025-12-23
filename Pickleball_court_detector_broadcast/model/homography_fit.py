# model/court_template.py
import numpy as np

def tennis_court_world():
    """
    ITF standard tennis court (singles)
    Units: meters
    Origin: bottom-left corner
    """

    points = {
        "BL": (0.0, 0.0),
        "BR": (8.23, 0.0),
        "TR": (8.23, 23.77),
        "TL": (0.0, 23.77),

        "service_left": (0.0, 11.885),
        "service_right": (8.23, 11.885),

        "center_bottom": (4.115, 0.0),
        "center_top": (4.115, 23.77),
    }

    lines = [
        ("BL", "BR"),
        ("BR", "TR"),
        ("TR", "TL"),
        ("TL", "BL"),
        ("service_left", "service_right"),
        ("center_bottom", "center_top"),
    ]

    return points, lines