import cv2
import numpy as np
import os
import math

# ================= PATHS =================
INPUT_VIDEO = r"C:\Users\Niranjan\OneDrive\Documents\video.mp4"
OUTPUT_DIR = r"C:\pickleball boundary\final_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_CAM = os.path.join(OUTPUT_DIR, "half_court_camera_view.mp4")
OUT_TOP = os.path.join(OUTPUT_DIR, "half_court_topdown_with_nvz.mp4")

# ================= VIDEO =================
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError("Cannot open input video")

fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_cam = cv2.VideoWriter(OUT_CAM, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

# Top-down (rectified) size
TOP_W, TOP_H = 600, 800
out_top = cv2.VideoWriter(OUT_TOP, cv2.VideoWriter_fourcc(*'mp4v'), fps, (TOP_W, TOP_H))

# ================= ROI =================
ROI_TOP = int(H * 0.60)
ROI_BOT = int(H * 0.95)

# ================= NVZ =================
NVZ_RATIO = 0.18  # ~7ft / 39ft (half-court length)

# ================= LINE HELPERS =================
def line_from_points(x1, y1, x2, y2):
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return a, b, c

def intersect(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    d = a1 * b2 - a2 * b1
    if abs(d) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / d
    y = (a2 * c1 - a1 * c2) / d
    return int(x), int(y)

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cam_view = frame.copy()
    topdown = np.zeros((TOP_H, TOP_W, 3), dtype=np.uint8)

    # ROI mask
    roi_mask = np.zeros((H, W), dtype=np.uint8)
    roi_mask[ROI_TOP:ROI_BOT, :] = 255

    # White line extraction (HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
    white = cv2.bitwise_and(white, roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=2)

    edges = cv2.Canny(white, 50, 150)

    # Hough
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, 80,
        minLineLength=80, maxLineGap=200
    )

    horizontals, slanted, verticals = [], [], []

    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
            length = math.hypot(x2-x1, y2-y1)

            if angle < 15:
                horizontals.append((x1,y1,x2,y2))
            elif 20 < angle < 70:
                slanted.append((x1,y1,x2,y2))
            elif 80 < angle < 100:
                verticals.append((x1,y1,x2,y2,length))

    if len(horizontals) < 2 or len(slanted) < 2:
        out_cam.write(cam_view)
        out_top.write(topdown)
        continue

    # Boundary lines
    horizontals.sort(key=lambda l:(l[1]+l[3])//2)
    slanted.sort(key=lambda l:min(l[0], l[2]))

    top_h = line_from_points(*horizontals[0])
    bot_h = line_from_points(*horizontals[-1])
    left_l = line_from_points(*slanted[0])
    right_l = line_from_points(*slanted[-1])

    TL = intersect(top_h, left_l)
    TR = intersect(top_h, right_l)
    BL = intersect(bot_h, left_l)
    BR = intersect(bot_h, right_l)

    if None in (TL, TR, BL, BR):
        out_cam.write(cam_view)
        out_top.write(topdown)
        continue

    # Draw boundary (camera view)
    cv2.polylines(cam_view, [np.array([TL, TR, BR, BL])], True, (0,0,255), 4)

    # ================= CENTER LINE (camera view) =================
    court_center_x = (TL[0] + TR[0] + BL[0] + BR[0]) / 4
    valid = [(x1,y1,x2,y2,l) for x1,y1,x2,y2,l in verticals
             if BL[0] < (x1+x2)/2 < BR[0]]

    if valid:
        valid.sort(key=lambda v:(-v[4], abs(((v[0]+v[2])/2)-court_center_x)))
        x1,y1,x2,y2,_ = valid[0]
        center_l = line_from_points(x1,y1,x2,y2)
        Ct = intersect(center_l, top_h)
        Cb = intersect(center_l, bot_h)
        if Ct and Cb:
            cv2.line(cam_view, Ct, Cb, (255,255,255), 3)

    # ================= HOMOGRAPHY =================
    src = np.float32([TL, TR, BR, BL])
    dst = np.float32([
        [0, 0],
        [TOP_W-1, 0],
        [TOP_W-1, TOP_H-1],
        [0, TOP_H-1]
    ])
    Hmat = cv2.getPerspectiveTransform(src, dst)
    topdown = cv2.warpPerspective(frame, Hmat, (TOP_W, TOP_H))

    # ================= DRAW STRAIGHT LINES (TOP-DOWN) =================
    # Outer boundary
    cv2.rectangle(topdown, (0,0), (TOP_W-1, TOP_H-1), (0,0,255), 3)

    # Center line (perfectly vertical)
    cv2.line(topdown, (TOP_W//2, 0), (TOP_W//2, TOP_H), (255,255,255), 3)

    # ================= NVZ (KITCHEN) =================
    nvz_y = int(NVZ_RATIO * TOP_H)

    # NVZ line
    cv2.line(topdown, (0, nvz_y), (TOP_W, nvz_y), (0,255,255), 3)

    # Optional shaded NVZ zone
    overlay = topdown.copy()
    cv2.rectangle(overlay, (0,0), (TOP_W, nvz_y), (0,255,255), -1)
    cv2.addWeighted(overlay, 0.15, topdown, 0.85, 0, topdown)

    # ================= OUTPUT =================
    out_cam.write(cam_view)
    out_top.write(topdown)

    cv2.imshow("Camera View", cam_view)
    cv2.imshow("Top-Down View (Straight + NVZ)", topdown)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out_cam.release()
out_top.release()
cv2.destroyAllWindows()

print("✅ Half court + center line + straightened court + NVZ generated")
print("📁 Camera view:", OUT_CAM)
print("📁 Top-down NVZ view:", OUT_TOP)
