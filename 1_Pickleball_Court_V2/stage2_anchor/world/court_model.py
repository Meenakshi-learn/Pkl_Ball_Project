# world/court_model.py
import numpy as np

COURT_WIDTH = 6.10
COURT_HALF_LENGTH = 6.70
NVZ = 2.13
CENTER_X = COURT_WIDTH / 2

def half_court():
    return np.array([
        [0, 0],
        [COURT_WIDTH, 0],
        [COURT_WIDTH, COURT_HALF_LENGTH],
        [0, COURT_HALF_LENGTH]
    ], np.float32)

def center_line():
    return np.array([
        [CENTER_X, 0],
        [CENTER_X, COURT_HALF_LENGTH]
    ], np.float32)

def nvz_line():
    return np.array([
        [0, NVZ],
        [COURT_WIDTH, NVZ]
    ], np.float32)