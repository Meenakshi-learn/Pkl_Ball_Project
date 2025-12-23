import cv2
import numpy as np

def extract_court_region(img):
    """
    Adaptive pickleball court region extraction.
    Robust for indoor courts with ceiling, banners, reflections.

    Returns:
        court_img  : image masked to court region
        court_mask : binary mask of court
    """

    H, W = img.shape[:2]

    # --------------------------------------------------
    # 1️⃣ Sample court color from bottom-center region
    # --------------------------------------------------
    # Physical assumption: court touches bottom of frame
    sample_h1 = int(H * 0.70)
    sample_h2 = int(H * 0.90)
    sample_w1 = int(W * 0.30)
    sample_w2 = int(W * 0.70)

    sample = img[sample_h1:sample_h2, sample_w1:sample_w2]

    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)

    # Robust statistics (median resists reflections)
    h_med, s_med, v_med = np.median(
        hsv.reshape(-1, 3), axis=0
    )

    # --------------------------------------------------
    # 2️⃣ Adaptive HSV bounds (pickleball-safe)
    # --------------------------------------------------
    lower = np.array([
        max(h_med - 15, 0),
        max(s_med - 60, 30),
        max(v_med - 80, 30)
    ], dtype=np.uint8)

    upper = np.array([
        min(h_med + 15, 180),
        255,
        255
    ], dtype=np.uint8)

    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_full, lower, upper)

    # --------------------------------------------------
    # 3️⃣ Morphological cleanup
    # --------------------------------------------------
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (25, 25)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # --------------------------------------------------
    # 4️⃣ Keep only bottom-connected components
    # (kills ceiling & banners completely)
    # --------------------------------------------------
    num_labels, labels = cv2.connectedComponents(mask)

    bottom_labels = set(
        labels[H-1, x] for x in range(W)
    )

    refined_mask = np.zeros_like(mask)
    for lbl in bottom_labels:
        refined_mask[labels == lbl] = 255

    # --------------------------------------------------
    # 5️⃣ Final court mask + image
    # --------------------------------------------------
    court_img = cv2.bitwise_and(img, img, mask=refined_mask)

    return court_img, refined_mask