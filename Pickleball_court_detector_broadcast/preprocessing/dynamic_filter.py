import cv2
import numpy as np

def dynamic_preprocess(frame):
    """
    Dynamic, distance-aware preprocessing for court detection.
    Returns: gray, filtered, edges
    """

    h, w = frame.shape[:2]

    # --------------------------------------------------
    # 1. Grayscale
    # --------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------
    # 2. Distance-aware bilateral filtering
    #    Near: preserve edges
    #    Far : stronger smoothing
    # --------------------------------------------------
    filtered = gray.copy()

    # Define horizon split (empirical)
    split_y = int(0.55 * h)

    # Near side (bottom)
    filtered[split_y:h, :] = cv2.bilateralFilter(
        gray[split_y:h, :],
        d=7,
        sigmaColor=40,
        sigmaSpace=40
    )

    # Far side (top)
    filtered[0:split_y, :] = cv2.bilateralFilter(
        gray[0:split_y, :],
        d=9,
        sigmaColor=90,
        sigmaSpace=90
    )

    # --------------------------------------------------
    # 3. Dynamic Canny thresholds (adaptive)
    # --------------------------------------------------
    med = np.median(filtered)
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))

    edges = cv2.Canny(filtered, lower, upper)

    # --------------------------------------------------
    # 4. Morphological stabilization
    # --------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    return gray, filtered, edges