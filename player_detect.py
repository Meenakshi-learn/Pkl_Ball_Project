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
COURT_WIDTH = 6.10     # left sideline to right sideline
HALF_COURT_LENGTH = 6.70  # baseline to net

# -----------------------------
# GPU CHECK AND SETUP
# -----------------------------
print("="*60)
print("JETSON NANO GPU STATUS CHECK")
print("="*60)

cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    device = "cuda:0"
else:
    print("⚠️  WARNING: CUDA not available, using CPU (will be slow!)")
    device = "cpu"

print(f"Using device: {device}")
print("=" * 60 + "\n")

# -----------------------------
# LOAD YOLO MODELS
# -----------------------------
print("Loading YOLO models...")
model = YOLO("yolov8n.pt")
pose_model = YOLO("yolov8n-pose.pt")

if cuda_available:
    model.to(device)
    pose_model.to(device)

print("✓ Models loaded")

# -----------------------------
# INPUT VIDEO
# -----------------------------
video_path = "C:/Users/1_HOME/2_Meenakshi/2_NOW_UNIVISION/pythonProject/Now_PKb_Project/videos/pickleball_court.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ ERROR: Could not open video!")
    exit()

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video: {W}x{H} @ {fps} fps\n")

# -----------------------------
# OUTPUT VIDEO
# -----------------------------
output_video_path = "C:/Users/1_HOME/2_Meenakshi/2_NOW_UNIVISION/pythonProject/Now_PKb_Project/result/output.mp4"
out = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (W, H)
)

# -----------------------------
# HALF COURT REGION
# -----------------------------
X_MIN = int(0.15 * W)
X_MAX = int(0.85 * W)
Y_NET = int(0.25 * H)
Y_BASELINE = int(0.80 * H)

half_court_polygon = Polygon([
    (X_MIN, Y_NET),
    (X_MAX, Y_NET),
    (X_MAX, Y_BASELINE),
    (X_MIN, Y_BASELINE)
])

# -----------------------------
# COURT REFERENCE POINTS (IMAGE SPACE)
# Manually verify these once using a frame
# -----------------------------
img_pts = np.array([
    [X_MIN, Y_BASELINE],   # Near-left baseline corner  (0,0)
    [X_MAX, Y_BASELINE],   # Near-right baseline corner (W,0)
    [X_MIN, Y_NET],        # Near-left net intersection (0,L)
    [X_MAX, Y_NET],        # Near-right net intersection (W,L)
], dtype=np.float32)

# -----------------------------
# COURT REFERENCE POINTS (COURT SPACE)
# -----------------------------
court_pts = np.array([
    [0.0, 0.0],
    [COURT_WIDTH, 0.0],
    [0.0, HALF_COURT_LENGTH],
    [COURT_WIDTH, HALF_COURT_LENGTH],
], dtype=np.float32)

# -----------------------------
# COMPUTE HOMOGRAPHY (IMAGE → COURT)
# -----------------------------
H, status = cv2.findHomography(img_pts, court_pts)


# -----------------------------
# TRACKING + KT
# -----------------------------
stable_id_counter = 1
player_history = {}
DISTANCE_THRESHOLD = 200
FRAME_MEMORY = 300

player1_kt = []           # (frame, foot_x, foot_y)
player_coordinates = []   # full data

# ---------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------
def get_bbox_features(x1, y1, x2, y2, frame):
    roi = frame[y1:y2, x1:x2]
    avg_color = roi.mean(axis=(0, 1)) if roi.size > 0 else np.array([0, 0, 0])
    return {
        "aspect_ratio": (y2 - y1) / (x2 - x1 + 1e-6),
        "color": avg_color
    }

# ---------------------------------------------------
# PLAYER MATCHING
# ---------------------------------------------------
def match_player(cx, cy, features, frame_no):
    global stable_id_counter

    expired = [
        sid for sid, info in player_history.items()
        if frame_no - info["last_frame"] > FRAME_MEMORY
    ]
    for sid in expired:
        del player_history[sid]

    if not player_history:
        player_history[1] = {
            "last_pos": (cx, cy),
            "last_frame": frame_no,
            "appearance": features,
        }
        return 1

    px, py = player_history[1]["last_pos"]
    d = math.hypot(cx - px, cy - py)

    player_history[1]["last_pos"] = (cx, cy)
    player_history[1]["last_frame"] = frame_no
    return 1

# ---------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------
frame_no = 0
print("▶ Processing video... Press Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1

    results = model.track(
        frame,
        conf=0.4,
        classes=[0],
        tracker="bytetrack.yaml",
        persist=True,
        device=device,
        verbose=False
    )

    pose_out = pose_model(frame, conf=0.4, device=device, verbose=False)
    kpts = pose_out[0].keypoints.data.cpu().numpy() if hasattr(pose_out[0], "keypoints") else None

    if results[0].boxes.xyxy is not None:
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if not half_court_polygon.contains(Point(cx, cy)):
                continue

            features = get_bbox_features(x1, y1, x2, y2, frame)
            sid = match_player(cx, cy, features, frame_no)

            if sid != 1:
                continue

            # ---- FOOT POINTS ----
            if kpts is not None:
                lfx, lfy = int(kpts[0][15][0]), int(kpts[0][15][1])
                rfx, rfy = int(kpts[0][16][0]), int(kpts[0][16][1])
            else:
                lfx, lfy = x1 + (x2 - x1)//4, y2
                rfx, rfy = x1 + 3*(x2 - x1)//4, y2

            if lfy >= rfy:
                kt_x, kt_y = lfx, lfy
            else:
                kt_x, kt_y = rfx, rfy
                            
            # -----------------------------
            # PROJECT KT TO COURT COORDINATES
            # -----------------------------
            pt_img = np.array([[[kt_x, kt_y]]], dtype=np.float32)
            pt_court = cv2.perspectiveTransform(pt_img, H)

            court_x = float(pt_court[0][0][0])
            court_y = float(pt_court[0][0][1])


            player1_kt.append((frame_no, court_x, court_y))

            cv2.circle(frame, (kt_x, kt_y), 3, (0, 0, 255), -1)
            cv2.putText(frame, "KT", (kt_x + 5, kt_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

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

# ---------------------------------------------------
# SAVE KT CSV
# ---------------------------------------------------
kt_csv_path = "C:/Users/1_HOME/2_Meenakshi/2_NOW_UNIVISION/pythonProject/Now_PKb_Project/result/player1_kt.csv"
with open(kt_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "court_x_m", "court_y_m"])
    writer.writerows(player1_kt)

print("📈 Player 1 KT saved at:", kt_csv_path)
print("✅ DONE")
