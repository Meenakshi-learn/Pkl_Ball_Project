import cv2
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Geometry
# --------------------------------------------------
from .geometry.line_classification import split_and_filter_lines
from .geometry.near_side_trapezoid import detect_near_side_trapezoid
from .geometry.trapezoid_tracker import TrapezoidTracker
from .geometry.weighted_hough import detect_weighted_hough_lines

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
from .preprocessing.dynamic_filter import dynamic_preprocess
from .preprocessing.line_confidence import compute_line_confidence

# --------------------------------------------------
# Video utils
# --------------------------------------------------
from Pickleball_court_detector_broadcast.utils.video_utils import (
    video_frame_generator,
    normalize_lighting
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE = Path(__file__).parent
VIDEO_PATH = BASE / "input" / "pkb_court.mp4"

STAGES = BASE / "stages"
FINAL  = BASE / "final"

STAGES.mkdir(exist_ok=True)
FINAL.mkdir(exist_ok=True)


def main():

    tracker = TrapezoidTracker(window=7)

    for fid, frame in video_frame_generator(VIDEO_PATH):

        # 1️⃣ Normalize lighting
        frame = normalize_lighting(frame)

        # 2️⃣ Dynamic preprocessing
        gray, filtered, edges = dynamic_preprocess(frame)

        # 3️⃣ Line-confidence heatmap
        confidence = compute_line_confidence(gray, edges)

        # 4️⃣ Guided Hough transform
        lines, weighted_edges = detect_weighted_hough_lines(edges, confidence)
        if lines is None:
            continue

        # Save representative stages once
        if fid == 0:
            cv2.imwrite(str(STAGES / "01_gray.jpg"), gray)
            cv2.imwrite(str(STAGES / "02_filtered.jpg"), filtered)
            cv2.imwrite(str(STAGES / "03_edges.jpg"), edges)
            cv2.imwrite(str(STAGES / "04_line_confidence.jpg"), confidence)
            cv2.imwrite(str(STAGES / "05_weighted_edges.jpg"), weighted_edges)

        lines = lines[:, 0, :]
        print(f"[DEBUG] Frame {fid} | lines: {lines.shape[0]}")

        # 5️⃣ Line classification (near-side)
        horiz, oblique = split_and_filter_lines(lines, frame.shape)
        if len(horiz) < 1 or len(oblique) < 2:
            continue

        # 6️⃣ Near-side trapezoid detection
        trapezoid = detect_near_side_trapezoid(horiz, oblique, frame.shape)
        tracker.update(trapezoid)

        stable = tracker.get_stable()
        if stable is not None:
            vis = frame.copy()
            cv2.polylines(vis, [stable], True, (0, 255, 0), 4)
            cv2.imwrite(str(FINAL / "near_side_trapezoid.png"), vis)

            print("✅ Stable near-side trapezoid detected")
            break


if __name__ == "__main__":
    main()