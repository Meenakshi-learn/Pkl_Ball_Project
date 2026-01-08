# utils/intersection_builder.py
import numpy as np

def intersect(l1, l2):
    x1,y1,x2,y2 = l1
    x3,y3,x4,y4 = l2

    denom = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None

    px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/denom
    py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/denom
    return int(px), int(py)


def build_intersections(horiz, vert):
    pts = []
    for h in horiz:
        for v in vert:
            p = intersect(h, v)
            if p:
                pts.append(p)
    return pts

def order_trapezoid_points(pts):
    """
    Orders trapezoid points as:
    BL, BR, TR, TL
    """
    pts = np.array(pts)

    if len(pts) != 4:
        return None

    # Sort by y (top to bottom)
    pts = pts[np.argsort(pts[:, 1])]

    top = pts[:2]
    bottom = pts[2:]

    TL = top[np.argmin(top[:, 0])]
    TR = top[np.argmax(top[:, 0])]

    BL = bottom[np.argmin(bottom[:, 0])]
    BR = bottom[np.argmax(bottom[:, 0])]

    return np.array([BL, BR, TR, TL], dtype=np.int32)