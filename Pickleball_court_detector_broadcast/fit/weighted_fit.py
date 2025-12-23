import cv2
import numpy as np


def fit_weighted_homography(img_pts, model_pts):
    """
    Fits homography using exactly 4 corresponding points.

    img_pts   : list of (x, y) image-space points
    model_pts : dict of world-space court points
    """

    if len(img_pts) < 4:
        return None

    # --------------------------------------------------
    # Select 4 extreme image points (court corners)
    # --------------------------------------------------
    img_pts = np.array(img_pts, dtype=np.float32)

    # Sort by y (top to bottom)
    img_pts = img_pts[np.argsort(img_pts[:, 1])]

    top = img_pts[:2]
    bottom = img_pts[-2:]

    # Sort left-right
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]

    img_corners = np.array([
        bottom[0],  # BL
        bottom[1],  # BR
        top[1],     # TR
        top[0],     # TL
    ], dtype=np.float32)

    # --------------------------------------------------
    # Corresponding WORLD points
    # --------------------------------------------------
    world_corners = np.array([
        model_pts["BL"],
        model_pts["BR"],
        model_pts["TR"],
        model_pts["TL"],
    ], dtype=np.float32)

    # --------------------------------------------------
    # Compute homography
    # --------------------------------------------------
    H, _ = cv2.findHomography(world_corners, img_corners, cv2.RANSAC)

    return H