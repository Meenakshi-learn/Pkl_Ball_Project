import cv2
import numpy as np
from Pickleball_court_detector_broadcast.model.pickleball_court_template import (
    pickleball_court_world
)

def draw_near_side_court(img, H, margin=0.5):
    """
    Draw outer boundary + pickleball lines on near side
    """

    if H is None:
        return img

    points, lines = pickleball_court_world()

    # ---- Outer boundary with margin
    outer = np.array([
        (-margin, -margin),
        (6.10 + margin, -margin),
        (6.10 + margin, 6.705 + margin),
        (-margin, 6.705 + margin)
    ], dtype=np.float32).reshape(-1, 1, 2)

    proj_outer = cv2.perspectiveTransform(outer, H).astype(int)
    cv2.polylines(img, [proj_outer], True, (0, 0, 255), 4)

    # ---- Internal lines (near side only)
    for a, b in lines:
        p1 = points[a]
        p2 = points[b]

        # Skip far-side lines
        if p1[1] > 6.705 or p2[1] > 6.705:
            continue

        seg = np.array([p1, p2], dtype=np.float32).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(seg, H).astype(int)

        cv2.line(img,
                 tuple(proj[0][0]),
                 tuple(proj[1][0]),
                 (255, 255, 255),
                 3)

    return img