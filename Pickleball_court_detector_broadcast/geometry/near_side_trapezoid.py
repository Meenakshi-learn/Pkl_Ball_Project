import numpy as np

def intersect(l1, l2):
    """Line intersection: l = (x1,y1,x2,y2)"""
    x1,y1,x2,y2 = l1
    x3,y3,x4,y4 = l2

    A = np.array([
        [x2-x1, x3-x4],
        [y2-y1, y3-y4]
    ], dtype=np.float32)

    B = np.array([
        x3-x1,
        y3-y1
    ], dtype=np.float32)

    if abs(np.linalg.det(A)) < 1e-6:
        return None

    t = np.linalg.solve(A, B)[0]
    px = x1 + t*(x2-x1)
    py = y1 + t*(y2-y1)
    return np.array([px, py], dtype=np.int32)


def detect_near_side_trapezoid(horiz, oblique, img_shape):
    """
    Geometry-enforced near-side trapezoid detection
    """

    h, w = img_shape[:2]

    # ---- Bottom baseline (lowest horizontal)
    baseline = max(horiz, key=lambda l: (l[1] + l[3]) / 2)

    # ---- Candidate sidelines: long + bottom-anchored
    candidates = []
    for l in oblique:
        y_avg = (l[1] + l[3]) / 2
        length = np.hypot(l[2]-l[0], l[3]-l[1])
        if y_avg > 0.55*h and length > 0.4*w:
            candidates.append(l)

    if len(candidates) < 2:
        return None

    # ---- Pick left & right by x-position
    left  = min(candidates, key=lambda l: min(l[0], l[2]))
    right = max(candidates, key=lambda l: max(l[0], l[2]))

    # ---- Compute top corners by intersection
    tl = intersect(left, baseline)
    tr = intersect(right, baseline)

    if tl is None or tr is None:
        return None

    # ---- Bottom corners (use baseline endpoints)
    bl = np.array([baseline[0], baseline[1]], dtype=np.int32)
    br = np.array([baseline[2], baseline[3]], dtype=np.int32)

    trapezoid = np.array([bl, br, tr, tl], dtype=np.int32)

    return trapezoid