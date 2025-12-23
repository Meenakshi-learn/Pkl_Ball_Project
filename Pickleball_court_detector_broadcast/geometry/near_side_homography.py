import cv2
import numpy as np

def intersect(l1, l2):
    """
    Compute intersection of two lines given as (x1,y1,x2,y2)
    """
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    A = np.array([
        [x2 - x1, x3 - x4],
        [y2 - y1, y3 - y4]
    ], dtype=np.float32)

    B = np.array([
        x3 - x1,
        y3 - y1
    ], dtype=np.float32)

    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return None

    t, _ = np.linalg.solve(A, B)
    xi = int(x1 + t * (x2 - x1))
    yi = int(y1 + t * (y2 - y1))
    return (xi, yi)


def fit_near_side_homography(baseline, left_sideline, right_sideline):
    """
    Fit homography using near-side pickleball court geometry
    """

    tl = intersect(left_sideline, baseline)
    tr = intersect(right_sideline, baseline)

    if tl is None or tr is None:
        return None, None

    # Image points (clockwise)
    img_pts = np.array([
        tl,                       # TL
        tr,                       # TR
        (baseline[2], baseline[3]),  # BR
        (baseline[0], baseline[1])   # BL
    ], dtype=np.float32)

    # World points (meters, near-side only)
    world_pts = np.array([
        [0.0, 6.705],     # TL
        [6.10, 6.705],    # TR
        [6.10, 0.0],      # BR
        [0.0, 0.0]        # BL
    ], dtype=np.float32)

    H, _ = cv2.findHomography(world_pts, img_pts, cv2.RANSAC, 5.0)
    return H, img_pts