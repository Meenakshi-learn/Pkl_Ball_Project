from court_detector.src.utils.math_tools import line_length

def classify_vertical_horizontal(lines):
    vertical = []
    horizontal = []

    for (x1, y1, x2, y2) in lines:
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx < dy:
            vertical.append((x1, y1, x2, y2))
        else:
            horizontal.append((x1, y1, x2, y2))

    return vertical, horizontal


def keep_2_longest(lines):
    if not lines:
        return []
    lines = sorted(lines, key=lambda L: line_length(*L), reverse=True)
    return lines[:2]