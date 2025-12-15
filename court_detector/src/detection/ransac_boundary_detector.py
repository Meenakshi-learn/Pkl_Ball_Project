import cv2
from court_detector.src.preprocessing.color_mask import mask_court_blue
from court_detector.src.preprocessing.edge_detector import get_edges_from_masked
from court_detector.src.detection.hough_lines import detect_hough_lines
from court_detector.src.detection.line_filter import classify_vertical_horizontal
from court_detector.src.utils.intersections import intersection_point
from court_detector.src.visualization.debug_draw import draw_lines, draw_points

from court_detector.src.detection.ransac_fit import (
    fit_ransac_from_segments,
    horizontal_line_to_endpoints,
    vertical_line_to_endpoints
)


def detect_boundary_ransac(img):
    """
    Detects outer court boundary using:
        - Hough Lines
        - Line classification (vertical / horizontal)
        - RANSAC line fitting
        - Intersections (4 corners)
    """

    H, W = img.shape[:2]

    # -----------------------------
    # 1. Court Mask (net ignored)
    # -----------------------------
    mask = mask_court_blue(img)
    court_only = cv2.bitwise_and(img, img, mask=mask)

    # -----------------------------
    # 2. Edges
    # -----------------------------
    edges = get_edges_from_masked(court_only)

    # -----------------------------
    # 3. Hough Lines
    # -----------------------------
    raw = detect_hough_lines(edges, min_length=W // 3)
    if raw is None:
        return None, None, None

    # -----------------------------
    # 4. Classify into vertical / horizontal
    # -----------------------------
    vertical, horizontal = classify_vertical_horizontal(raw)

    if len(vertical) < 2 or len(horizontal) < 2:
        print("Not enough line segments for RANSAC.")
        return None, None, None

    # -----------------------------
    # 5. RANSAC Fitting
    # -----------------------------
    hv = fit_ransac_from_segments(horizontal, vertical=False)
    vv = fit_ransac_from_segments(vertical, vertical=True)

    if hv is None or vv is None:
        return None, None, None

    # RANSAC returns slope m and intercept b
    m_h, b_h = hv
    m_v, b_v = vv

    # Convert infinite lines to screen endpoints
    h_line = horizontal_line_to_endpoints(m_h, b_h, W)
    v_line = vertical_line_to_endpoints(m_v, b_v, H)

    # -----------------------------
    # 6. Compute FOUR outer corners
    # -----------------------------
    corners = []

    # top horizontal intersects both verticals
    corners.append(intersection_point(h_line, v_line))

    return h_line, v_line, corners