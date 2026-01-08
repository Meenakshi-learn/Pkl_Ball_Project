import numpy as np

def snap_line_to_edges(p1, p2, edge_img, search_radius=5, samples=40):
    """
    Snap a projected line to strong nearby edges by shifting
    perpendicular to the line direction.
    """

    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)

    # Direction of the line
    d = p2 - p1
    norm = np.linalg.norm(d)
    if norm < 1e-6:
        return tuple(p1.astype(int)), tuple(p2.astype(int))

    d /= norm

    # Perpendicular direction
    n = np.array([-d[1], d[0]])

    shifts = []

    for t in np.linspace(0.1, 0.9, samples):
        base = p1 + t * (p2 - p1)

        best_shift = 0
        best_val = 0

        for s in range(-search_radius, search_radius + 1):
            probe = base + s * n
            x, y = int(probe[0]), int(probe[1])

            if 0 <= y < edge_img.shape[0] and 0 <= x < edge_img.shape[1]:
                val = edge_img[y, x]
                if val > best_val:
                    best_val = val
                    best_shift = s

        shifts.append(best_shift)

    if len(shifts) == 0:
        return tuple(p1.astype(int)), tuple(p2.astype(int))

    avg_shift = np.mean(shifts)

    p1_ref = p1 + avg_shift * n
    p2_ref = p2 + avg_shift * n

    return tuple(p1_ref.astype(int)), tuple(p2_ref.astype(int))
