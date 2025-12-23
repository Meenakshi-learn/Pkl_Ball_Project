import cv2
import numpy as np


def video_frame_generator(video_path):
    """
    Generator that yields (frame_id, frame) from a video file
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame_id, frame
        frame_id += 1

    cap.release()


def normalize_lighting(frame):
    """
    Fix lighting, reflections, and mild noise for court surfaces
    """
    # LAB color space for illumination correction
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Preserve edges while denoising
    corrected = cv2.bilateralFilter(corrected, 9, 75, 75)

    return corrected


def draw_court_boundary(img, img_points):
    """
    Draw a convex hull boundary around detected court intersections
    """
    pts = np.array(img_points, dtype=np.int32)
    hull = cv2.convexHull(pts)

    overlay = img.copy()
    cv2.polylines(overlay, [hull], True, (0, 255, 0), 3)

    return overlay