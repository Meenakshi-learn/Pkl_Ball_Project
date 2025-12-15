import cv2
import numpy as np

def convex_boundary(points):
    pts = np.array(points, dtype="int32")
    hull = cv2.convexHull(pts)
    return hull.reshape(-1,2).tolist()