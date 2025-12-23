# court_detector_broadcast/fit/fit_court.py
import cv2
import numpy as np

def fit_court_homography(line_map, model_points):
    """
    Fit homography between detected court image points and world model points
    """

    # 🔹 TEMPORARY: Use image corners as detected points
    h, w = line_map.shape[:2]

    img_pts = np.array([
        [0, h],      # bottom-left
        [w, h],      # bottom-right
        [w, 0],      # top-right
        [0, 0],      # top-left
    ], dtype=np.float32)

    # 🔹 Corresponding world points (ORDER MATTERS)
    world_pts = np.array([
        model_points["BL"],
        model_points["BR"],
        model_points["TR"],
        model_points["TL"],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(world_pts, img_pts, cv2.RANSAC)

    return H