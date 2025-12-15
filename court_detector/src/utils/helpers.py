import cv2
import numpy as np
import os
import json

def ensure_dir(path):
    """Create a folder if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def save_image(path, img):
    """Save an image safely."""
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def save_json(path, data):
    """Save JSON output."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def sort_points_clockwise(points):
    """Sort (x,y) points clockwise."""
    if not points:
        return []

    pts = np.array(points)
    center = pts.mean(axis=0)

    def angle(p):
        return np.arctan2(p[1] - center[1], p[0] - center[0])

    pts_sorted = sorted(points, key=angle)
    return pts_sorted