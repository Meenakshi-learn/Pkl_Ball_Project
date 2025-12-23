import cv2
import numpy as np

def detect_near_side_boundary(frame):
    """
    Detect ONLY the near-side pickleball court trapezoid.
    Strategy:
    - Restrict to bottom half of image
    - Detect strong edges
    - Use convex hull of dominant edges
    """

    h, w = frame.shape[:2]

    # 🔒 Focus ONLY on near-side region
    roi = frame[int(0.55 * h):h, 0:w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    

    # Strong contrast for white lines
    edges = cv2.Canny(gray, 60, 160)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    pts = np.column_stack(np.where(edges > 0))

    if len(pts) < 500:
        return None

    # Convert (row, col) → (x, y)
    pts = np.array([(p[1], p[0]) for p in pts], dtype=np.int32)

    hull = cv2.convexHull(pts)

    # Shift hull back to full image coords
    hull[:, 0, 1] += int(0.55 * h)

    return hull