import cv2
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Geometry
# --------------------------------------------------
from geometry.hough_detector import detect_hough_lines
from geometry.line_classification import split_and_filter_lines
from geometry.select_sideline import select_sideline

from geometry.court_constants import (
    COURT_HALF_LENGTH,
    COURT_WIDTH,
    NVZ_DISTANCE,
    CENTER_X
)

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
from preprocessing.dynamic_filter import dynamic_preprocess

# --------------------------------------------------
# Utils
# --------------------------------------------------
from utils.video_utils import (
    video_frame_generator,
    normalize_lighting
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE = Path(__file__).parent
VIDEO_PATH = BASE / "input" / "pickleball_court.mp4"
FINAL = BASE / "final"
FINAL.mkdir(exist_ok=True)

# --------------------------------------------------
# Geometry helpers
# --------------------------------------------------
def fit_line(points):
    pts = np.array(points, dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(
        pts, cv2.DIST_L2, 0, 0.01, 0.01
    )
    return float(vx), float(vy), float(x0), float(y0)

def intersect_lines(l1, l2):
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2

    A = np.array([
        [vx1, -vx2],
        [vy1, -vy2]
    ], dtype=np.float32)

    B = np.array([x2 - x1, y2 - y1], dtype=np.float32)

    if abs(np.linalg.det(A)) < 1e-6:
        return None

    t = np.linalg.solve(A, B)[0]
    return int(x1 + t * vx1), int(y1 + t * vy1)

def project(world_pts, H):
    pts = np.array(world_pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).astype(int)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():

    H_frozen = None

    # -------- Lock output size --------
    gen0 = video_frame_generator(VIDEO_PATH)
    _, f0 = next(gen0)
    f0 = normalize_lighting(f0)
    h0, w0 = f0.shape[:2]
    roi0 = f0[int(0.35 * h0):h0, :]
    OUT_H, OUT_W = roi0.shape[:2]

    out = cv2.VideoWriter(
        str(FINAL / "near_side_court_output.avi"),
        cv2.VideoWriter_fourcc(*"MJPG"),
        25,
        (OUT_W, OUT_H),
        True
    )

    # -------- Main loop --------
    for fid, frame in video_frame_generator(VIDEO_PATH):

        frame = normalize_lighting(frame)
        h, w = frame.shape[:2]
        roi = frame[int(0.35 * h):h, :]

        _, _, edges = dynamic_preprocess(roi)

        lines = detect_hough_lines(edges)
        if lines is None:
            continue

        lines = lines[:, 0, :]
        horiz, oblique = split_and_filter_lines(lines, roi.shape)

        if len(horiz) < 2 or len(oblique) < 2:
            continue

        # -------- Select segments --------
        baseline_seg = max(horiz, key=lambda l: (l[1] + l[3]) / 2)

        left_seg = select_sideline(oblique, baseline_seg, None)
        mirrored = [[w - l[0], l[1], w - l[2], l[3]] for l in oblique]
        right_m = select_sideline(mirrored, baseline_seg, None)

        if left_seg is None or right_m is None:
            continue

        right_seg = [
            w - right_m[0],
            right_m[1],
            w - right_m[2],
            right_m[3]
        ]

        # -------- Fit infinite lines --------
        baseline = fit_line([(baseline_seg[0], baseline_seg[1]),
                              (baseline_seg[2], baseline_seg[3])])

        left = fit_line([(left_seg[0], left_seg[1]),
                         (left_seg[2], left_seg[3])])

        right = fit_line([(right_seg[0], right_seg[1]),
                          (right_seg[2], right_seg[3])])

        # -------- NVZ-based top line (parallel) --------
        vx, vy, x0, y0 = baseline
        offset = -NVZ_DISTANCE * 40  # empirical but stable
        top = (vx, vy, x0, y0 + offset)

        # -------- Rectangle corners --------
        BL = intersect_lines(left, baseline)
        BR = intersect_lines(right, baseline)
        TL = intersect_lines(left, top)
        TR = intersect_lines(right, top)

        if None in (BL, BR, TL, TR):
            continue

        image_pts = np.array([BL, BR, TR, TL], np.float32)

        # -------- Homography (freeze once) --------
        if H_frozen is None:
            world_pts = np.array([
                [0, 0],
                [COURT_WIDTH, 0],
                [COURT_WIDTH, COURT_HALF_LENGTH],
                [0, COURT_HALF_LENGTH]
            ], np.float32)

            H, _ = cv2.findHomography(world_pts, image_pts, cv2.RANSAC, 3.0)
            if H is not None:
                H_frozen = H

        # -------- Draw aligned court --------
        vis = roi.copy()

        if H_frozen is not None:
            court = project([
                (0, 0),
                (COURT_WIDTH, 0),
                (COURT_WIDTH, COURT_HALF_LENGTH),
                (0, COURT_HALF_LENGTH)
            ], H_frozen)

            cv2.polylines(
                vis,
                [court],
                True,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            center = project([
                (CENTER_X, 0),
                (CENTER_X, COURT_HALF_LENGTH)
            ], H_frozen)

            cv2.line(
                vis,
                tuple(center[0][0]),
                tuple(center[1][0]),
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )

        out.write(vis)
        print("Writing frame", fid)

    out.release()
    cv2.destroyAllWindows()
    print("✅ Stable half-court detection complete")

if __name__ == "__main__":
    main()
