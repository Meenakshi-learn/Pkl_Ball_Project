import os
import cv2

# ---------------- IMPORTS ----------------
from court_detector.src.io.loader import load_image
from court_detector.src.io.saver import save_image, save_json

from court_detector.src.preprocessing.color_mask import mask_court_blue
from court_detector.src.preprocessing.edge_detector import get_edges_from_masked

from court_detector.src.detection.hough_lines import detect_hough_lines
from court_detector.src.detection.line_filter import (
    classify_vertical_horizontal,
    keep_2_longest
)

from court_detector.src.detection.boundary_points import compute_outer_boundary
from court_detector.src.visualization.debug_draw import draw_lines, draw_points


# ---------------- PATHS ----------------
INPUT = "court_detector/src/input/court.jpg"
STAGE_DIR = "court_detector/src/output/stages/"
FINAL_DIR = "court_detector/src/output/final/"

os.makedirs(STAGE_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)


# ---------------- MAIN PIPELINE ----------------
def main():
    print("🚀 Running Outer Boundary Detector...")

    img = load_image(INPUT)
    H, W = img.shape[:2]

    # -------- 1. Mask Court --------
    mask = mask_court_blue(img)
    save_image(STAGE_DIR + "1_mask.jpg", mask)

    # -------- 2. Edge Detection --------
    court_only = cv2.bitwise_and(img, img, mask=mask)
    edges = get_edges_from_masked(court_only)
    save_image(STAGE_DIR + "2_edges.jpg", edges)

    # -------- 3. Hough Lines --------
    raw_lines = detect_hough_lines(edges, min_length=W // 3)
    save_image(STAGE_DIR + "3_raw_lines.jpg",
               draw_lines(img, raw_lines, (0, 0, 255)))

    # -------- 4. Vertical / Horizontal --------
    vertical, horizontal = classify_vertical_horizontal(raw_lines)

    vertical = keep_2_longest(vertical)
    horizontal = keep_2_longest(horizontal)

    save_image(
        STAGE_DIR + "4_vh_filtered.jpg",
        draw_lines(draw_lines(img, vertical, (0, 255, 0)),
                   horizontal, (255, 0, 0))
    )

    # -------- 5. No RANSAC for now --------
    V_final, H_final = vertical, horizontal

    # -------- 6. Outer Boundary Intersections --------
    outer_pts = compute_outer_boundary(V_final, H_final)

    save_image(FINAL_DIR + "outer_points.jpg",
               draw_points(img, outer_pts))

    save_json(FINAL_DIR + "outer_points.json",
              {"points": outer_pts})

    print("✔ DONE — Outer boundary + intersection points extracted successfully.")


# ---------------- RUN SCRIPT ----------------
if __name__ == "__main__":
    main()
