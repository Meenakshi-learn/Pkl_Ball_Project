import cv2
import numpy as np

def dynamic_preprocess(frame):
    """
    Dynamic, distance-aware preprocessing for court detection.

    Returns:
        gray_raw     : unfiltered grayscale (for NVZ / Weber boost)
        filtered     : smoothed grayscale (for strong lines)
        edges        : edges from filtered image (baseline + sidelines)
    """

    h, w = frame.shape[:2]

    # --------------------------------------------------
    # 1️⃣ RAW Grayscale (DO NOT TOUCH — NVZ depends on this)
    # --------------------------------------------------
    gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------
    # 2️⃣ Distance-aware bilateral filtering (ONLY for strong lines)
    # --------------------------------------------------
    filtered = gray_raw.copy()

    split_y = int(0.55 * h)

    # Near side (preserve court lines)
    filtered[split_y:h, :] = cv2.bilateralFilter(
        gray_raw[split_y:h, :],
        d=5,
        sigmaColor=25,
        sigmaSpace=25
    )

    # Far side (aggressive smoothing)
    filtered[0:split_y, :] = cv2.bilateralFilter(
        gray_raw[0:split_y, :],
        d=9,
        sigmaColor=90,
        sigmaSpace=90
    )

    # --------------------------------------------------
    # 3️⃣ Adaptive Canny (ONLY for strong edges)
    # --------------------------------------------------
    med = np.median(filtered)
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))

    edges = cv2.Canny(filtered, lower, upper)

    # --------------------------------------------------
    # 4️⃣ VERY light morphology (avoid thick noise)
    # --------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.dilate(edges, kernel, iterations=1)

    return gray_raw, filtered, edges