# court_detector_broadcast/visualization/draw.py
import cv2
import numpy as np

def draw_projected_court(img, H, model_points):
    """
    Projects world-space court template onto image
    """

    overlay = img.copy()

    # 🔹 Select and ORDER points explicitly
    ordered_keys = ["BL", "BR", "TR", "TL"]
    pts_world = np.array(
        [model_points[k] for k in ordered_keys],
        dtype=np.float32
    ).reshape(-1, 1, 2)

    # 🔹 Project points
    pts_img = cv2.perspectiveTransform(pts_world, H)

    # 🔹 Draw polygon
    cv2.polylines(
        overlay,
        [pts_img.astype(int)],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3
    )

    return overlay