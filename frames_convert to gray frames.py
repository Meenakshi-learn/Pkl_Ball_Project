import cv2
import numpy as np
import os
import shutil

# ---------------- PATHS -------------------
input_folder  = r"C:\boundry\ROI_Frames"
sharp_folder  = r"C:\boundry\Sharp_Gray_Frames"
blur_folder   = r"C:\boundry\Blur_Frames"  # optional

# Create folders if not existing
os.makedirs(sharp_folder, exist_ok=True)
os.makedirs(blur_folder, exist_ok=True)

# ----------- ROI FUNCTION -----------------
def get_roi_mask(frame):
    h, w = frame.shape[:2]

    # CURRENTLY FULL FRAME ROI (safe default)
    polygon = np.array([[(0,0), (w,0), (w,h), (0,h)]], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, polygon, 255)
    return mask

# ----------- BLUR CHECK -------------------
def is_blurry(gray_image, threshold=100):
    """Detect blur using Laplacian variance"""
    lap_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    return lap_var < threshold

# ------------- PROCESS EACH FRAME ----------
for file in sorted(os.listdir(input_folder)):

    if not (file.lower().endswith(".jpg") or file.lower().endswith(".png")):
        continue

    img_path = os.path.join(input_folder, file)
    frame = cv2.imread(img_path)

    if frame is None:
        print("❌ Cannot read:", file)
        continue

    # ----------- APPLY ROI ----------------
    mask = get_roi_mask(frame)
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    # ----------- GRAYSCALE ----------------
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # ----------- BLUR FILTERING -----------
    if is_blurry(gray, threshold=120):   # tune threshold if needed
        shutil.copy(img_path, os.path.join(blur_folder, file))
        print(f"❌ BLUR IMAGE REMOVED: {file}")
        continue

    # ----------- SAVE SHARP GRAY ----------
    save_path = os.path.join(sharp_folder, file)
    cv2.imwrite(save_path, gray)
    print(f"✔ SAVED SHARP GRAY: {file}")

print("\n🎉 DONE — Grayscale frames saved & blurry frames removed!")
