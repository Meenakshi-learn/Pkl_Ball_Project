import cv2
import numpy as np
import os

def isolate_white_lines(input_image_path, output_path):

    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Image not found:", input_image_path)
        return

    # Detect bright white lines
    _, line_mask = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

    # Dilate to strengthen line continuity
    kernel = np.ones((5, 5), np.uint8)
    line_mask = cv2.dilate(line_mask, kernel, iterations=1)

    # Blur entire frame
    blurred = cv2.GaussianBlur(img, (25, 25), 0)

    # Convert to 3 channel
    line_mask_3 = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR)
    img_3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    blurred_3 = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    # Keep only sharp lines
    sharp_lines = cv2.bitwise_and(img_3, line_mask_3)

    # Blur background outside lines
    inverted_mask = cv2.bitwise_not(line_mask_3)
    blurred_background = cv2.bitwise_and(blurred_3, inverted_mask)

    # Combine sharp + blurred
    final = cv2.add(sharp_lines, blurred_background)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final)

    print(f"✔ Saved: {output_path}")


# =====================================================
# PROCESS THE ENTIRE DENOISED FRAME FOLDER
# =====================================================

input_folder  = r"C:\Users\Niranjan\court line tracking and extraction\output_frames\denoised"
output_folder = r"C:\Users\Niranjan\court line tracking and extraction\output_frames\isolated_lines"

os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):

        input_path  = os.path.join(input_folder, file_name)
        frame_name  = os.path.splitext(file_name)[0]

        output_path = os.path.join(output_folder, f"{frame_name}_isolated.jpg")

        isolate_white_lines(input_path, output_path)

print("\n🎉 BATCH WHITE-LINE ISOLATION COMPLETE ✔")
