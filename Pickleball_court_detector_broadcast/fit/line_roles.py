import numpy as np

def classify_pickleball_lines(lines, img_shape):
    """
    Classify pickleball court lines based on geometry.
    Returns a dict of role → line.
    """

    h, w = img_shape[:2]

    horizontals = []
    verticals = []

    # --------------------------------------------------
    # Split by orientation
    # --------------------------------------------------
    for x1, y1, x2, y2 in lines:
        if abs(y1 - y2) < abs(x1 - x2):
            horizontals.append((x1, y1, x2, y2))
        else:
            verticals.append((x1, y1, x2, y2))

    # --------------------------------------------------
    # Sort by position
    # --------------------------------------------------
    horizontals.sort(key=lambda l: (l[1] + l[3]) / 2)
    verticals.sort(key=lambda l: (l[0] + l[2]) / 2)

    roles = {}

    # --------------------------------------------------
    # Assign roles (pickleball logic)
    # --------------------------------------------------
    if len(horizontals) >= 2:
        roles["baseline_bottom"] = horizontals[0]
        roles["baseline_top"] = horizontals[-1]

    if len(horizontals) >= 3:
        roles["nvz_bottom"] = horizontals[1]
        roles["nvz_top"] = horizontals[-2]

    if len(verticals) >= 2:
        roles["sideline_left"] = verticals[0]
        roles["sideline_right"] = verticals[-1]

    if len(verticals) >= 3:
        roles["center_line"] = verticals[len(verticals)//2]

    return roles