import numpy as np
import cv2

def snap_line_to_white(p1, p2, edges, search=8):
    p1 = np.array(p1, np.float32)
    p2 = np.array(p2, np.float32)

    d = p2 - p1
    d /= (np.linalg.norm(d) + 1e-6)
    n = np.array([-d[1], d[0]])

    best_score = -1
    best_shift = 0

    for s in range(-search, search + 1):
        score = 0
        shift = s * n

        for t in np.linspace(0.05, 0.95, 60):
            pt = p1 + t * (p2 - p1) + shift
            x, y = int(pt[0]), int(pt[1])

            if 0 <= y < edges.shape[0] and 0 <= x < edges.shape[1]:
                score += 1 if edges[y, x] > 0 else 0

        if score > best_score:
            best_score = score
            best_shift = s

    return p1 + best_shift * n, p2 + best_shift * n

def valid_segment(p1, p2, min_len=40):
    """
    Reject degenerate or very short snapped segments
    """
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    return np.linalg.norm(p1 - p2) > min_len

def center_line_in_band(p1, p2, edges, search=4):
    """
    Shift line slightly along normal to center it
    inside the white paint band.
    """
    p1 = np.array(p1, np.float32)
    p2 = np.array(p2, np.float32)

    d = p2 - p1
    d /= (np.linalg.norm(d) + 1e-6)
    n = np.array([-d[1], d[0]])  # normal

    best_shift = 0
    best_balance = 1e9

    for s in range(-search, search + 1):
        shift = s * n
        white_count = 0
        gray_count = 0

        for t in np.linspace(0.1, 0.9, 30):
            pt = p1 + t * (p2 - p1) + shift
            x, y = int(pt[0]), int(pt[1])

            if 0 <= y < edges.shape[0] and 0 <= x < edges.shape[1]:
                if edges[y, x] > 0:
                    white_count += 1
                else:
                    gray_count += 1

        balance = abs(white_count - gray_count)
        if balance < best_balance:
            best_balance = balance
            best_shift = s

    return p1 + best_shift * n, p2 + best_shift * n

def refine_line_center(p1, p2, edges, max_shift=4):
    """
    Fine-centers a line inside a thick white band by testing
    small normal shifts and choosing the most symmetric response.
    """
    p1 = np.array(p1, np.float32)
    p2 = np.array(p2, np.float32)

    d = p2 - p1
    d /= (np.linalg.norm(d) + 1e-6)
    n = np.array([-d[1], d[0]])

    best_shift = 0
    best_score = -1

    for s in range(-max_shift, max_shift + 1):
        shift = s * n
        score = 0

        for t in np.linspace(0.1, 0.9, 60):
            pt = p1 + t * (p2 - p1) + shift
            x, y = int(pt[0]), int(pt[1])

            if 0 <= y < edges.shape[0] and 0 <= x < edges.shape[1]:
                score += int(edges[y, x])

        if score > best_score:
            best_score = score
            best_shift = s

    return p1 + best_shift * n, p2 + best_shift * n

def snap_projected_line(line_pts, edges, search=6):
    """
    line_pts: (p1, p2) in image space (float)
    edges: binary or strong-edge image
    """
    p1, p2 = np.array(line_pts[0]), np.array(line_pts[1])

    d = p2 - p1
    d = d / (np.linalg.norm(d) + 1e-6)
    n = np.array([-d[1], d[0]])  # normal direction

    best_shift = 0
    best_score = -1

    for s in np.linspace(-search, search, 2*search+1):
        score = 0
        shift = s * n

        for t in np.linspace(0.1, 0.9, 50):
            pt = p1 + t*(p2-p1) + shift
            x, y = int(pt[0]), int(pt[1])
            if 0 <= y < edges.shape[0] and 0 <= x < edges.shape[1]:
                score += edges[y, x]

        if score > best_score:
            best_score = score
            best_shift = s

    p1_refined = p1 + best_shift * n
    p2_refined = p2 + best_shift * n

    return p1_refined, p2_refined

def refine_line_subpixel(p1, p2, edges, band=2):
    pts = []
    d = p2 - p1
    d = d / (np.linalg.norm(d) + 1e-6)
    n = np.array([-d[1], d[0]])

    for t in np.linspace(0.05, 0.95, 80):
        base = p1 + t*(p2-p1)
        for s in range(-band, band+1):
            pt = base + s*n
            x, y = int(pt[0]), int(pt[1])
            if 0 <= y < edges.shape[0] and 0 <= x < edges.shape[1]:
                if edges[y, x] > 0:
                    pts.append(pt)

    if len(pts) < 20:
        return p1, p2

    vx, vy, x0, y0 = cv2.fitLine(
        np.array(pts, dtype=np.float32),
        cv2.DIST_L2, 0, 0.01, 0.01
    )

    p1_new = np.array([x0, y0]) - 1000*np.array([vx, vy])
    p2_new = np.array([x0, y0]) + 1000*np.array([vx, vy])
    return p1_new, p2_new
