import cv2
import numpy as np

def hull_to_corners(hull, epsilon_ratio=0.02):
    """
    Reduce convex hull to 4 corner points (court trapezoid)
    """
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon_ratio * peri, True)

    if approx is None or len(approx) != 4:
        return None

    return approx.reshape(-1, 2)