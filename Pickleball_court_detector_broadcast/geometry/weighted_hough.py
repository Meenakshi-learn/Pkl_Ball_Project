import cv2
import numpy as np

def detect_weighted_hough_lines(edges, confidence):
    """
    Bias Hough voting using line confidence map.
    """

    # Normalize confidence
    conf = confidence.astype(np.float32) / 255.0

    # Weight edges
    weighted_edges = (edges * conf).astype(np.uint8)

    lines = cv2.HoughLinesP(
        weighted_edges,
        rho=1,
        theta=np.pi/180,
        threshold=80,
        minLineLength=80,
        maxLineGap=20
    )

    return lines, weighted_edges