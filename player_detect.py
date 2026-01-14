from ultralytics import YOLO
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
import csv
import math
import warnings
import torch
warnings.filterwarnings("ignore")

# -----------------------------
# STANDARD PICKLEBALL COURT (meters)
# -----------------------------
COURT_WIDTH = 6.10
HALF_COURT_LENGTH = 6.70

# -----------------------------
# GPU CHECK AND SETUP
# -----------------------------
print("="*60)
print("JETSON NANO GPU STATUS CHECK")
print("="*60)

cuda_available = torch.cuda.is_available()
device = "cuda:0" if cuda_available else "cpu"
print(f"CUDA Available: {cuda_available}")
print(f"Using device: {device}")
print("="*60 + "\n")

# -----------------------------
# LOAD YOLO MODELS
# -----------------------------
model = YOLO("yolov8n.pt")
pose_model = YOLO("yolov8n-pose.pt")
if cuda_available:
    model.to(device)
    pose_model.to(device)

# -----------------------------
# INPUT VIDEO
# -----------------------------
video_path = "C:/Users/1_HOME/2_Meenakshi/2_NOW_UNIVISION/pythonProject/Now_PKb_Project/videos/pickleball_court.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Could not open video")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# -----------------------------
# OUTPUT VIDEO
# -----------------------------
output_video_path = "C:/Users/1_HOME/2_Meenakshi/2_NOW_UNIVISION/pythonProject/Now_PKb_Project/result/output.mp4"
out = cv2.VideoWriter(output_video_path,
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps, (W, H))

# -----------------------------
# HALF COURT REGION
# -----------------------------
X_MIN, X_MAX = int(0.15 * W), int(0.85 * W)
Y_NET, Y_BASELINE = int(0.25 * H), int(0.80 * H)

half_court_polygon = Polygon([
    (X_MIN, Y_NET),
    (X_MAX, Y_NET),
    (X_MAX, Y_BASELINE),
    (X_MIN, Y_BASELINE)
])

# -----------------------------
# HOMOGRAPHY SETUP
# -----------------------------
img_pts = np.array([
    [X_MIN, Y_BASELINE],
    [X_MAX, Y_BASELINE],
    [X_MIN, Y_NET],
    [X_MAX, Y_NET],
], dtype=np.float32)

court_pts = np.array([
    [0.0, 0.0],
    [COURT_WIDTH, 0.0],
    [0.0, HALF_COURT_LENGTH],
    [COURT_WIDTH, HALF_COURT_LENGTH],
], dtype=np.float32)

H_mat, _ = cv2.findHomography(img_pts, court_pts)
if H_mat is None:
    raise RuntimeError("Homography computation failed")

# -----------------------------
# TRACKING DATA
# -----------------------------
player1_left_kt = []
player1_right_kt = []
player_history = {}

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def get_bbox_features(x1, y1, x2, y2, frame):
    roi = frame[y1:y2, x1:x2]
    avg_color = roi.mean(axis=(0, 1)) if roi.size > 0 else np.zeros(3)
    return {"color": avg_color}

# -----------------------------
# PLAYER MATCHING (SINGLE PLAYER)
# -----------------------------
def match_player(cx, cy, features, frame_no):
    if not player_history:
        player_history[1] = {"last_pos": (cx, cy), "last_frame": frame_no}
    else:
        player_history[1]["last_pos"] = (cx, cy)
        player_history[1]["last_frame"] = frame_no
    return 1

# -----------------------------
# MAIN LOOP
# -----------------------------
frame_no = 0
print("▶ Processing video... Press Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1

    results = model.track(frame, conf=0.4, classes=[0],
                          tracker="bytetrack.yaml",
                          persist=True, device=device, verbose=False)

    pose_out = pose_model(frame, conf=0.4, device=device, verbose=False)
    kpts = pose_out[0].keypoints.data.cpu().numpy() if hasattr(pose_out[0], "keypoints") else None
    if kpts is None or len(kpts) == 0:
        continue

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if not half_court_polygon.contains(Point(cx, cy)):
            continue

        sid = match_player(cx, cy, {}, frame_no)
        if sid != 1:
            continue

        # Left & Right ankles
        lfx, lfy = int(kpts[0][15][0]), int(kpts[0][15][1])
        rfx, rfy = int(kpts[0][16][0]), int(kpts[0][16][1])

        # Project LEFT
        left_pt = cv2.perspectiveTransform(
            np.array([[[lfx, lfy]]], dtype=np.float32), H_mat)
        lx, ly = map(float, left_pt[0][0])
        player1_left_kt.append((frame_no, lx, ly))

        # Project RIGHT
        right_pt = cv2.perspectiveTransform(
            np.array([[[rfx, rfy]]], dtype=np.float32), H_mat)
        rx, ry = map(float, right_pt[0][0])
        player1_right_kt.append((frame_no, rx, ry))

        # Draw
        cv2.circle(frame, (lfx, lfy), 4, (0, 0, 255), -1)
        cv2.circle(frame, (rfx, rfy), 4, (255, 0, 0), -1)
        cv2.putText(frame, "L", (lfx + 4, lfy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(frame, "R", (rfx + 4, rfy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "ID 1", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Processed Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# -----------------------------
# SAVE CSVs
# -----------------------------
base = "C:/Users/1_HOME/2_Meenakshi/2_NOW_UNIVISION/pythonProject/Now_PKb_Project/result"

with open(f"{base}/player1_left_kt.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "court_x_m", "court_y_m"])
    writer.writerows(player1_left_kt)

with open(f"{base}/player1_right_kt.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "court_x_m", "court_y_m"])
    writer.writerows(player1_right_kt)

print("✅ DONE – Bilateral KT saved")
