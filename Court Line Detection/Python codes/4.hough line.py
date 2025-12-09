import cv2
import numpy as np
import os

def process_edges_and_hough(input_path, edges_folder, hough_folder, frame_name):

    img = cv2.imread(input_path)
    if img is None:
        print("❌ Cannot read:", input_path)
        return

    # -------- GRAY + BLUR --------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # -------- CANNY EDGE DETECTION --------
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    cv2.imwrite(
        os.path.join(edges_folder, f"{frame_name}_edges.jpg"),
        edges
    )

    # -------- HOUGH LINE DETECTION --------
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=80,
        maxLineGap=20
    )

    line_img = img.copy()

    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(
        os.path.join(hough_folder, f"{frame_name}_hough.jpg"),
        line_img
    )

    print(f"✔ Done: {frame_name}")


# ==============================================================
# BATCH PROCESSING
# ==============================================================

input_folder = r"C:\Users\Niranjan\court line tracking and extraction\output_frames\only_court_lines"

output_base   = r"C:\Users\Niranjan\court line tracking and extraction\output_frames"

edges_folder  = os.path.join(output_base, "edges")
hough_folder  = os.path.join(output_base, "hough_lines")

os.makedirs(edges_folder, exist_ok=True)
os.makedirs(hough_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):

        frame_path = os.path.join(input_folder, file_name)
        frame_name = os.path.splitext(file_name)[0]

        process_edges_and_hough(
            input_path=frame_path,
            edges_folder=edges_folder,
            hough_folder=hough_folder,
            frame_name=frame_name
        )

print("\n🎉 BATCH COMPLETE — EDGE + HOUGH SAVED IN SEPARATE FOLDERS ✔")
