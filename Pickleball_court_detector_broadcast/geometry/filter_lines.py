# geometry/filter_lines.py
import numpy as np

def filter_court_candidate_lines(lines):
    """
    Remove near-horizontal / background lines.
    Keeps steep perspective court edges.
    """
    if lines is None:
        return None

    filtered = []

    for x1, y1, x2, y2 in lines[:, 0]:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

        # Indoor broadcast court constraints
        if 20 < angle < 70 or 110 < angle < 160:
            filtered.append([x1, y1, x2, y2])

    if len(filtered) < 4:
        return None

    return np.array(filtered).reshape(-1, 1, 4)