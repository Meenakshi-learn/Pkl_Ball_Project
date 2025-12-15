import cv2
import numpy as np

def mask_court_blue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([85, 40, 40])
    upper_blue = np.array([135, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean = np.zeros_like(mask)
    clean[labels == largest] = 255

    return clean
