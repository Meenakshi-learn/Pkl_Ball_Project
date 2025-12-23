import cv2
import numpy as np

def extract_white_lines(roi):
    """
    Extract white court lines from near-side ROI.
    """

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # White lines: low saturation, high value
    lower = np.array([0, 0, 180])
    upper = np.array([180, 70, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Strengthen lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask