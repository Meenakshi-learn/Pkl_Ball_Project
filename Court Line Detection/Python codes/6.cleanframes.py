import cv2
import numpy as np
import os

input_folder  = r"C:\Users\Niranjan\court line tracking and extraction\output_frames\manual_court_lines"
output_folder = r"C:\Users\Niranjan\court line tracking and extraction\output_frames\clean_manual"

os.makedirs(output_folder, exist_ok=True)

def clean_one_image(img_path, save_path):
    global points
    points = []

    img = cv2.imread(img_path)
    clone = img.copy()

    def click_event(event, x, y, flags, param):
        global points, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                cv2.circle(clone, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(
                    clone, str(len(points)),
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2
                )
                cv2.imshow("Click 1,3,10,12", clone)

    cv2.imshow("Click 1,3,10,12", clone)
    cv2.setMouseCallback("Click 1,3,10,12", click_event)

    print("\n📌 Click ORDER = P1, P3, P10, P12")
    print("👉 After clicking 4 points, press ENTER")

    # wait until ENTER
    while True:
        key = cv2.waitKey(1)
        if key == 13:   # ENTER key
            break

    cv2.destroyAllWindows()

    # validation
    if len(points) != 4:
        print("❌ You must click exactly 4 points.")
        return

    P1, P3, P10, P12 = points

    # polygon
    h, w = img.shape[:2]
    outer_polygon = np.array([P1, P3, P10, P12], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [outer_polygon], 255)

    blurred = cv2.GaussianBlur(img, (41, 41), 0)

    clean = np.where(mask[..., None] == 255, img, blurred)

    cv2.imwrite(save_path, clean)
    print("✔ SAVED:", save_path)


# ======================================================
# BATCH LOOP
# ======================================================

for file in os.listdir(input_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):

        img_path  = os.path.join(input_folder, file)
        save_name = os.path.splitext(file)[0] + "_clean.jpg"
        save_path = os.path.join(output_folder, save_name)

        print("\n=========================================")
        print("🖼  PROCESSING:", file)
        print("=========================================")

        clean_one_image(img_path, save_path)

print("\n🎉 ALL MANUAL COURT CLEANING COMPLETED ✔")
