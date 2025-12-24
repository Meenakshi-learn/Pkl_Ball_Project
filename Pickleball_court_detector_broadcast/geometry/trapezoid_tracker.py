import numpy as np
from collections import deque

class TrapezoidTracker:
    def __init__(self, window=7):
        self.buffer = deque(maxlen=window)

    def update(self, trapezoid):
        if trapezoid is not None:
            self.buffer.append(trapezoid)

    def get_stable(self):
        if len(self.buffer) < 3:
            return None
        return np.median(np.stack(self.buffer), axis=0).astype(np.int32)