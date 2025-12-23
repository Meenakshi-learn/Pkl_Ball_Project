import numpy as np

def line_intersection(l1, l2):
    """
    Compute intersection of two lines in (x1,y1,x2,y2) format.
    Returns (x,y) or None.
    """
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None

    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom

    return np.array([px, py], dtype=np.float32)


def extract_intersections(horizontal_lines, vertical_lines):
    """
    Intersect each horizontal with each vertical line.
    """
    points = []

    for h in horizontal_lines:
        for v in vertical_lines:
            p = line_intersection(h, v)
            if p is not None:
                points.append(p)

    return np.array(points)

def filter_intersections(points, image_shape):
    """
    Keep intersections inside image bounds.
    """
    h, w = image_shape[:2]

    valid = []
    for p in points:
        x, y = p
        if 0 <= x < w and 0 <= y < h:
            valid.append(p)

    return np.array(valid)