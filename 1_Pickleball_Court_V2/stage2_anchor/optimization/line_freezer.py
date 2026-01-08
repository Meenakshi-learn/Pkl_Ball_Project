import numpy as np

class LineFreezer:
    """
    Generic temporal freezer for:
    - lines: shape (N, 4)
    - polygons: shape (N, 2)
    """

    def __init__(self, anchor_frames=12):
        self.anchor_frames = anchor_frames
        self.buffer = []
        self.frozen = None

    def update(self, item):
        """
        item: np.ndarray
        """

        if self.frozen is not None:
            return self.frozen

        self.buffer.append(item)

        if len(self.buffer) < self.anchor_frames:
            return None

        self.frozen = np.mean(
            np.stack(self.buffer, axis=0),
            axis=0
        )

        print("✅ Polygon frozen")
        return self.frozen