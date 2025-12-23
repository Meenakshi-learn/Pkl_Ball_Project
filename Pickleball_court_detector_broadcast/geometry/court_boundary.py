import cv2
import numpy as np

def boundary_from_hough_lines(lines):
    """
    Compute court boundary as convex hull of Hough line endpoints
    """
    if lines is None or len(lines) == 0:
        return None

    pts = []
    for x1, y1, x2, y2 in lines[:, 0]:
        pts.append([x1, y1])
        pts.append([x2, y2])

    pts = np.array(pts, dtype=np.int32)

    if len(pts) < 6:
        return None

    return cv2.convexHull(pts)

def boundary_from_intersections(points, img_shape):
    """
    Robust court boundary from intersection cluster
    """
    if points is None or len(points) < 6:
        return None

    pts = np.array(points, dtype=np.int32)

    # Remove extreme top background (ceiling)
    h = img_shape[0]
    pts = pts[pts[:, 1] > h * 0.35]

    if len(pts) < 6:
        return None

    hull = cv2.convexHull(pts)

    # Approximate to quadrilateral
    eps = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, eps, True)

    return approx if len(approx) >= 4 else hull