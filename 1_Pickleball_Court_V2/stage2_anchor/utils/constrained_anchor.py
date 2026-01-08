import numpy as np
import cv2


def line_normal(p1, p2):
    """Unit normal vector of line p1->p2"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    n = np.array([-dy, dx], dtype=np.float32)
    norm = np.linalg.norm(n)
    return n / norm if norm > 0 else n


def estimate_edge_offset(p1, p2, edges, search=6, samples=40):
    """
    Estimate perpendicular offset for a line using edge responses.
    Returns scalar offset (can be + or -).
    """
    h, w = edges.shape
    normal = line_normal(p1, p2)

    offsets = []
    for t in np.linspace(0.1, 0.9, samples):
        x = int(p1[0] + t * (p2[0] - p1[0]))
        y = int(p1[1] + t * (p2[1] - p1[1]))

        best = None
        best_mag = 0

        for d in range(-search, search + 1):
            xs = int(x + d * normal[0])
            ys = int(y + d * normal[1])

            if 0 <= xs < w and 0 <= ys < h:
                mag = edges[ys, xs]
                if mag > best_mag:
                    best_mag = mag
                    best = d

        if best is not None:
            offsets.append(best)

    if len(offsets) < samples * 0.3:
        return 0.0  # insufficient support → no correction

    return float(np.mean(offsets))


def shift_line(p1, p2, offset):
    """Shift line p1-p2 by offset along its normal"""
    n = line_normal(p1, p2)
    shift = offset * n
    p1s = (p1[0] + shift[0], p1[1] + shift[1])
    p2s = (p2[0] + shift[0], p2[1] + shift[1])
    return p1s, p2s


def intersect_lines(p1, p2, p3, p4):
    """Intersection of two infinite lines"""
    A = np.array([
        [p2[0] - p1[0], p3[0] - p4[0]],
        [p2[1] - p1[1], p3[1] - p4[1]]
    ], dtype=np.float32)

    B = np.array([
        p3[0] - p1[0],
        p3[1] - p1[1]
    ], dtype=np.float32)

    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return None

    t = np.linalg.solve(A, B)[0]
    return (p1[0] + t * (p2[0] - p1[0]),
            p1[1] + t * (p2[1] - p1[1]))