import cv2
import numpy as np

def suppress_players(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, fg = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

    suppressed = img.copy()
    suppressed[fg > 0] = (0, 0, 0)

    return suppressed