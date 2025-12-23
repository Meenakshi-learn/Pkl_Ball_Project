import cv2
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Geometry
# --------------------------------------------------
from .geometry.hough_detector import detect_hough_lines
from .geometry.line_classification import split_and_filter_lines
from .geometry.near_side_homography import fit_near_side_homography
from .geometry.project_template import project_near_side_template

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
from .preprocessing.reinforce_lines import reinforce_lines

# --------------------------------------------------
# Video utils
# --------------------------------------------------
from Pickleball_court_detector_broadcast.utils.video_utils import (
    video_frame_generator,
    normalize_lighting
)

BASE = Path(__file__).parent
VIDEO_PATH = BASE / "input" / "pkb_court.mp4"
STAGES = BASE / "stages"
FINAL  = BASE / "final"

STAGES.mkdir(exist_ok=True)
FINAL.mkdir(exist_ok=True)


def main():
    # --------------------------------------------------
    # 0️⃣ Select frame
    # --------------------------------------------------
    GOOD_FRAME_ID = 40
    frame = None

    for fid, f in video_frame_generator(VIDEO_PATH):
        if fid == GOOD_FRAME_ID:
            frame = f
            break

    if frame is None:
        raise RuntimeError("Good frame not found")

    frame = normalize_lighting(frame)
    cv2.imwrite(str(STAGES / "00_frame.jpg"), frame)

    # --------------------------------------------------
    # 1️⃣ Dynamic preprocessing
    # --------------------------------------------------
    from .preprocessing.dynamic_filter import dynamic_preprocess

    gray, filtered, edges = dynamic_preprocess(frame)

    cv2.imwrite(str(STAGES / "01_gray.jpg"), gray)
    cv2.imwrite(str(STAGES / "02_filtered.jpg"), filtered)
    cv2.imwrite(str(STAGES / "03_edges.jpg"), edges)

    # --------------------------------------------------
    # 3️⃣ Hough lines
    # --------------------------------------------------
    lines = detect_hough_lines(edges)
    if lines is None:
        raise RuntimeError("No lines detected")

    lines = lines[:, 0, :]
    print("[DEBUG] lines:", lines.shape)

    # --------------------------------------------------
    # 4️⃣ Line classification
    # --------------------------------------------------
    horiz, oblique = split_and_filter_lines(lines, frame.shape)

    if len(horiz) < 1 or len(oblique) < 2:
        raise RuntimeError("Insufficient near-side lines")

    # Baseline = lowest horizontal
    baseline = max(horiz, key=lambda l: (l[1] + l[3]) / 2)

    # Side lines
    left_sideline  = min(oblique, key=lambda l: min(l[0], l[2]))
    right_sideline = max(oblique, key=lambda l: max(l[0], l[2]))

    # --------------------------------------------------
    # 5️⃣ Debug visualization
    # --------------------------------------------------
    dbg = frame.copy()
    cv2.line(dbg, baseline[:2], baseline[2:], (0, 0, 255), 4)
    cv2.line(dbg, left_sideline[:2], left_sideline[2:], (255, 0, 0), 4)
    cv2.line(dbg, right_sideline[:2], right_sideline[2:], (255, 0, 0), 4)
    cv2.imwrite(str(STAGES / "03_selected_lines.jpg"), dbg)

    # --------------------------------------------------
    # 6️⃣ Homography
    # --------------------------------------------------
    H, _ = fit_near_side_homography(
        baseline,
        left_sideline,
        right_sideline
    )

    if H is None:
        raise RuntimeError("Near-side homography failed")

    # --------------------------------------------------
    # 7️⃣ Project stabilized court
    # --------------------------------------------------
    result = project_near_side_template(frame, H)
    cv2.imwrite(str(FINAL / "near_side_court.png"), result)

    print("✅ Near-side pickleball court detected & aligned")


if __name__ == "__main__":
    main()