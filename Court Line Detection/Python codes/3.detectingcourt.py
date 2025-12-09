import cv2
import numpy as np
import os

def isolate_white_lines_high_contrast(input_img_path, output_img_path):

    gray = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print("❌ Image not found:", input_img_path)
        return

    # ---- detect white lines ----
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # ---- improve continuity ----
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ---- dark background completely ----
    background = (gray * 0.12).astype(np.uint8)  # 12% brightness

    # ---- overlay sharp white court lines ----
    final = background.copy()
    final[mask == 255] = 255

    # ---- save output ----
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    cv2.imwrite(output_img_path, final)

    print(f"✔ Saved: {output_img_path}")


# ===================================================================
# BATCH PROCESSOR  (YOUR REQUEST)
# ===================================================================

input_folder  = r"C:\Users\Niranjan\court line tracking and extraction\output_frames\isolated_lines"
output_folder = r"C:\Users\Niranjan\court line tracking and extraction\output_frames\only_court_lines"

os.makedirs(output_folder, exist_ok=True)

# Loop all files
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):

        input_path  = os.path.join(input_folder, file_name)
        frame_name  = os.path.splitext(file_name)[0]

        output_path = os.path.join(output_folder, f"{frame_name}_onlylines.jpg")

        isolate_white_lines_high_contrast(input_path, output_path)

print("\n🎉 BATCH PROCESS COMPLETE: Only court lines extracted ✔")
