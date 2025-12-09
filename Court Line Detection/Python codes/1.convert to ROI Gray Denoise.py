import cv2
import numpy as np
import os

def process_full_court(input_image_path, roi_folder, gray_folder, denoise_folder, frame_name):

    img = cv2.imread(input_image_path)
    if img is None:
        print(f"❌ ERROR reading image: {input_image_path}")
        return

    orig = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Detect blue court
    lower_blue = np.array([85, 30, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((25, 25), np.uint8)
    mask_clean = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print(f"⚠ Court not detected in: {frame_name}")
        return

    all_points = np.vstack([cnt for cnt in contours])
    x, y, w, h = cv2.boundingRect(all_points)

    # ROI + clean outputs
    roi = orig[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # SAVE in different folders
    cv2.imwrite(os.path.join(roi_folder, f"{frame_name}_roi.jpg"), roi)
    cv2.imwrite(os.path.join(gray_folder, f"{frame_name}_gray.jpg"), gray)
    cv2.imwrite(os.path.join(denoise_folder, f"{frame_name}_denoised.jpg"), denoised)

    print(f"✅ Saved ROI, Gray, Denoised for: {frame_name}")


# ====================== FOLDER SETTINGS ======================
input_folder  = r"C:\Users\Niranjan\court line tracking and extraction\frames"
output_base   = r"C:\Users\Niranjan\court line tracking and extraction\output_frames"

roi_folder     = os.path.join(output_base, "roi")
gray_folder    = os.path.join(output_base, "gray")
denoise_folder = os.path.join(output_base, "denoised")

os.makedirs(roi_folder, exist_ok=True)
os.makedirs(gray_folder, exist_ok=True)
os.makedirs(denoise_folder, exist_ok=True)

# ====================== PROCESS ENTIRE FOLDER ======================
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):

        frame_path = os.path.join(input_folder, file_name)
        frame_name = os.path.splitext(file_name)[0]

        process_full_court(
            input_image_path=frame_path,
            roi_folder=roi_folder,
            gray_folder=gray_folder,
            denoise_folder=denoise_folder,
            frame_name=frame_name
        )

print("\n🎉 BATCH PROCESS COMPLETED ✔")
