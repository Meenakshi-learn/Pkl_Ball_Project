import numpy as np
from Pickleball_court_detector_broadcast.utils.intersection_builder import intersect


def select_sideline(oblique_lines, baseline, nvz_line=None):
    """
    Select the most plausible sideline.
    - Geometry-only if nvz_line is None
    - NVZ-refined if nvz_line is available
    """

    # --------------------------------------------------
    # Geometry-only mode (NO NVZ)
    # --------------------------------------------------
    if nvz_line is None:
        bx = (baseline[0] + baseline[2]) / 2

        def score(l):
            p = intersect(l, baseline)
            if p is None:
                return -np.inf
            return abs(p[0] - bx)

        return max(oblique_lines, key=score)

    # --------------------------------------------------
    # NVZ-refined mode
    # --------------------------------------------------
    best = None
    best_score = -np.inf

    for l in oblique_lines:
        p1 = intersect(l, baseline)
        p2 = intersect(l, [*nvz_line[0], *nvz_line[1]])

        if p1 is None or p2 is None:
            continue

        # prefer lines spanning more vertical distance
        score = abs(p2[1] - p1[1])

        if score > best_score:
            best_score = score
            best = l

    return best
