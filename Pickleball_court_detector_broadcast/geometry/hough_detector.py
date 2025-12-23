import cv2
import numpy as np

def detect_hough_lines(edge_img):
    """
    Detect straight lines using probabilistic Hough transform
    Tuned for indoor pickleball courts
    """
    lines = cv2.HoughLinesP(
        edge_img,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=60,
        maxLineGap=40
    )
    return lines