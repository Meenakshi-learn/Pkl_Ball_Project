import cv2
import numpy as np

def detect_hough_lines(edges, min_length):
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=120,
        minLineLength=min_length,
        maxLineGap=40
    )
    if raw is None:
        return []
    return [tuple(l[0]) for l in raw]
