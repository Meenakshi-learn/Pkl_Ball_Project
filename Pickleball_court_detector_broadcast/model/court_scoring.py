# model/court_scoring.py
import numpy as np

def reprojection_error(projected, detected):
    """
    Mean L2 distance between projected template points
    and detected image points
    """
    projected = np.array(projected)
    detected = np.array(detected)

    return np.mean(np.linalg.norm(projected - detected, axis=1))


def is_valid_court(error, thresh=15):
    """
    Error threshold in pixels
    """
    return error < thresh