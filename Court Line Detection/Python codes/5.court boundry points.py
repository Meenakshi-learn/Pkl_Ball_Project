import cv2
import os

# ==============================================
# SETTINGS
# ==============================================

input_folder  = r"C:\Users\Niranjan\court line tracking and extraction\output_frames\hough_lines"
output_folder = r"C:\Users\Niranjan\court line tracking and extraction\output_frames\manual_court_lines"

os.makedirs(output_folder, exist_ok=True)

names = ["1","2","3","6","5","4","7","8","9","12","11","10"]


# ==============================================
# FUNCTION
# ==============================================

def process_one_image(img_path):

    img = cv2.imread(img_path)
    if img is None:
        print("❌ Cannot read:", img_path)
        return

    clone = img.copy()
    points = []

    def mouse_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = len(points)
            if idx < 12:
                points.append((x, y))
                cv2.circle(clone, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(clone, names[idx], (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("CLICK 12 POINTS", clone)

    print("\n📌 CLICK ORDER (VERY IMPORTANT):")
    print("BOTTOM HALF = 1,2,3,6,5,4")
    print("TOP HALF    = 7,8,9,12,11,10")
    print("Press ENTER when finished...\n")

    cv2.imshow("CLICK 12 POINTS", clone)
    cv2.setMouseCallback("CLICK 12 POINTS", mouse_event)

    while True:
        if cv2.waitKey(1) == 13:  # ENTER key
            break

    cv2.destroyAllWindows()

    if len(points) != 12:
        print("❌ ERROR: You must click exactly 12 points")
        return

    # Unpack
    P1, P2, P3, P6, P5, P4, P7, P8, P9, P12, P11, P10 = points

    output = img.copy()
    color = (0,255,0)

    # ===== DRAW BOTTOM HALF =====
    cv2.line(output, P1, P2, color, 3)
    cv2.line(output, P2, P3, color, 3)
    cv2.line(output, P6, P5, color, 3)
    cv2.line(output, P5, P4, color, 3)
    cv2.line(output, P1, P6, color, 3)
    cv2.line(output, P2, P5, color, 3)
    cv2.line(output, P3, P4, color, 3)

    # ===== DRAW TOP HALF =====
    cv2.line(output, P12, P11, color, 3)
    cv2.line(output, P11, P10, color, 3)
    cv2.line(output, P7, P8, color, 3)
    cv2.line(output, P8, P9, color, 3)
    cv2.line(output, P12, P7, color, 3)
    cv2.line(output, P11, P8, color, 3)
    cv2.line(output, P10, P9, color, 3)

    # ===== CONNECT HALVES =====
    cv2.line(output, P6, P7, color, 3)
    cv2.line(output, P4, P9, color, 3)

    # LABEL POINTS
    labels = {
        "1":P1,"2":P2,"3":P3,"4":P4,"5":P5,"6":P6,
        "7":P7,"8":P8,"9":P9,"10":P10,"11":P11,"12":P12
    }

    for name,(x,y) in labels.items():
        cv2.circle(output, (x,y), 7, (0,0,255), -1)
        cv2.putText(output, name, (x+5,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Save result
    fname = os.path.basename(img_path)
    save_path = os.path.join(output_folder, fname.replace(".jpg","_court.jpg"))

    cv2.imwrite(save_path, output)

    print("\n✅ SAVED:", save_path)


# ==============================================
# MAIN LOOP OVER ALL IMAGES
# ==============================================

for file in os.listdir(input_folder):
    if file.lower().endswith((".jpg",".jpeg",".png",".bmp")):

        full_path = os.path.join(input_folder, file)

        print("\n=======================================")
        print("📌 Processing:", file)
        print("=======================================")

        process_one_image(full_path)

print("\n🎉 ALL IMAGES COMPLETED MANUALLY ✔")
