from court_detector.src.detection.intersections import intersection

def compute_outer_boundary(vertical, horizontal, diagonal):
    all_lines = vertical + horizontal + diagonal
    points = []

    for i in range(len(all_lines)):
        for j in range(i+1, len(all_lines)):
            inter = intersection(all_lines[i], all_lines[j])
            if inter:
                points.append(inter)

    # Remove duplicates
    points = list(set(points))

    # Sort by y then x
    points = sorted(points, key=lambda p: (p[1], p[0]))

    return points
