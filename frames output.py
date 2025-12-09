import cv2
import os
import numpy as np

# ------------------ PATHS ------------------------
input_folder = r"C:\boundry\Sharp_Gray_Frames"
output_folder = r"C:\boundry\framesoutput"

# -------------------------------------------------
# Create output directory if not existing
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ------------------ ROI MASK FUNCTION -----------
def get_roi_mask(frame):
    h, w = frame.shape[:2]

    # ***** EDIT THIS POLYGON FOR YOUR COURT AREA *****
    # Example ROI = full frame (safe default)
    polygon = np.array([[(0,0), (w,0), (w,h), (0,h)]], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, polygon, 255)
    return mask

# ------------------ PROCESS FRAMES --------------

count = 0

for file in sorted(os.listdir(input_folder)):

    # process only image files
    if not (file.lower().endswith(".jpg") or file.lower().endswith(".png")):
        continue

    img_path = os.path.join(input_folder, file)
    frame = cv2.imread(img_path)

    if frame is None:
        print("❌ Skipped:", file)
        continue

    # ---------------- ROI ------------------------
    mask = get_roi_mask(frame)
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    # ---------------- GRAYSCALE --------------------
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # ---------------- GAUSSIAN BLUR ----------------
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)

    # ---------------- CANNY ------------------------
    edges = cv2.Canny(blur, 50, 150)

    # ---------------- MORPHOLOGY -------------------
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # ---------------- HOUGH ------------------------
    lines = cv2.HoughLinesP(
        clean,
        rho=1,
        theta=np.pi/180,
        threshold=80,
        minLineLength=50,
        maxLineGap=10
    )

    output = frame.copy()

    if lines is not None:
      for line in lines:
         x1, y1, x2, y2 = line[0]   # <-- FIX
         cv2.line(output, (x1, y1), (x2, y2), (0,255,0), 2)

    # ---------------- SAVE FRAME -------------------
    save_path = os.path.join(output_folder, f"frame_{count:05}.jpg")
    cv2.imwrite(save_path, output)
    count += 1

    print("✔ Saved:", save_path)

print("\n🎉 DONE — all frames processed and saved!")