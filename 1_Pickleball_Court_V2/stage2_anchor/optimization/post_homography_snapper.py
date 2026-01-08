import numpy as np

def snap_poly_to_white(poly, edges, search_radius=4):
    """
    Snap projected polygon points slightly
    toward strongest nearby edge responses.
    """
    snapped = []

    for (x, y) in poly.reshape(-1, 2):
        best = (x, y)
        best_score = 0

        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                xx, yy = int(x + dx), int(y + dy)
                if 0 <= yy < edges.shape[0] and 0 <= xx < edges.shape[1]:
                    score = edges[yy, xx]
                    if score > best_score:
                        best_score = score
                        best = (xx, yy)

        snapped.append(best)

    return np.array(snapped, dtype=np.int32)