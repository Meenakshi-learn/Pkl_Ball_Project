import cv2
import numpy as np

def extract_court_region(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Broad green/blue court range (broadcast-safe)
    lower = np.array([30, 30, 30])
    upper = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    court = cv2.bitwise_and(img, img, mask=mask)
    return court, mask