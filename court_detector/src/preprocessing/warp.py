import cv2
import numpy as np

def warp_perspective(img, H, size):
    w, h = size
    return cv2.warpPerspective(img, H, (w, h))