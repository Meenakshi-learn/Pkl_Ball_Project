import cv2
import numpy as np

def project_near_side_template(img, H):
    """
    Project near-side pickleball court using real-world dimensions.
    """

    # Near-side pickleball court (meters)
    court = np.float32([
        [0.0, 0.0],        # Bottom-left
        [6.10, 0.0],       # Bottom-right
        [6.10, 6.705],     # Top-right (half court)
        [0.0, 6.705]       # Top-left
    ]).reshape(-1, 1, 2)

    # Map world → image
    proj = cv2.perspectiveTransform(court, H).astype(int)

    # Draw outer boundary
    cv2.polylines(img, [proj], True, (0, 0, 255), 4)

    return img