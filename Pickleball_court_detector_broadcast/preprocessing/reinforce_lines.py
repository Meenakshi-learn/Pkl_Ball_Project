import cv2
import numpy as np

def reinforce_lines(img):
    """
    Court-line–biased edge extraction using:
    - Bilateral filtering (edge-preserving denoise)
    - Adaptive thresholding (dynamic illumination handling)
    - Canny + dilation for Hough stability
    """

    # 1️⃣ Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2️⃣ Bilateral filter (preserve thin white lines)
    gray = cv2.bilateralFilter(
        gray,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )

    # 3️⃣ Adaptive threshold (handles far-side lighting loss)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=-5
    )

    # 4️⃣ Edge detection
    edges = cv2.Canny(thresh, 50, 150)

    # 5️⃣ Strengthen long court lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    return edges