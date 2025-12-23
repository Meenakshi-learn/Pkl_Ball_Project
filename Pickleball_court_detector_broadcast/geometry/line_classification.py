import numpy as np

def split_and_filter_lines(lines, img_shape):
    """
    Near-side pickleball line classification (position + geometry based).
    Robust to perspective and glare.
    """

    h, w = img_shape[:2]

    horizontals = []
    obliques = []

    # --------------------------------------------------
    # Dynamic minimum length (scales with resolution)
    # --------------------------------------------------
    min_len = 0.12 * max(h, w)

    def line_length(l):
        return np.hypot(l[2] - l[0], l[3] - l[1])

    for x1, y1, x2, y2 in lines:
        length = line_length((x1, y1, x2, y2))
        if length < min_len:
            continue

        dx = x2 - x1
        dy = y2 - y1
        angle = abs(np.degrees(np.arctan2(dy, dx)))

        y_mid = (y1 + y2) / 2
        x_mid = (x1 + x2) / 2

        # --------------------------------------------------
        # Horizontal lines (baseline / service line)
        # --------------------------------------------------
        if angle < 25 or angle > 155:
            # Keep only near-side horizontals
            if y_mid > 0.60 * h:
                horizontals.append((x1, y1, x2, y2))

        # --------------------------------------------------
        # Oblique lines (left & right sidelines)
        # --------------------------------------------------
        else:
            # Prefer lines spanning significant vertical range
            if abs(dy) > 0.15 * h:
                obliques.append((x1, y1, x2, y2))

    return horizontals, obliques