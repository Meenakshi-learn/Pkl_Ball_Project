import cv2
import numpy as np

def compute_line_confidence(gray, edges):
    """
    Produce a soft line-confidence heatmap (0–255)
    """

    h, w = gray.shape

    # --- Gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)

    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # --- Spatial prior (near-side stronger)
    y_coords = np.arange(h).reshape(-1, 1)
    spatial_weight = (y_coords / h) ** 1.5   # emphasize bottom
    spatial_weight = spatial_weight.astype(np.float32)

    confidence = magnitude * spatial_weight
    confidence = cv2.normalize(confidence, None, 0, 255, cv2.NORM_MINMAX)

    confidence = confidence.astype(np.uint8)

    return confidence