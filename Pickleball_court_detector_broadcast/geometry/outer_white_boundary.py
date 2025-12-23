import cv2
import numpy as np

def detect_outer_white_boundary(white_mask):
    """
    Detect near-side outer court boundary from white line mask.
    """

    lines = cv2.HoughLinesP(
        white_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=150,
        maxLineGap=40
    )

    if lines is None:
        return None

    # Flatten
    lines = lines[:, 0, :]

    candidates = []
    for x1, y1, x2, y2 in lines:
        length = np.hypot(x2 - x1, y2 - y1)
        if length > 200:   # keep only strong boundary lines
            candidates.append((x1, y1, x2, y2))

    if len(candidates) < 4:
        return None

    # Collect endpoints
    pts = []
    for l in candidates:
        pts.append((l[0], l[1]))
        pts.append((l[2], l[3]))

    pts = np.array(pts, dtype=np.int32)

    hull = cv2.convexHull(pts)
    return hull