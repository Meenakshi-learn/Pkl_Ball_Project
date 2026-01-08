import cv2
import numpy as np

def detect_hough_lines(edge_img):
    """
    Detect straight lines using probabilistic Hough transform
    Tuned for near-side pickleball court + NVZ
    """

    h, w = edge_img.shape[:2]

    lines = cv2.HoughLinesP(
        edge_img,
        rho=1,
        theta=np.pi / 180,
        threshold=60,                # ↓ allow faint lines
        minLineLength=35,            # ↓ NVZ is SHORT
        maxLineGap=60                # ↑ NVZ is BROKEN
    )

    return lines