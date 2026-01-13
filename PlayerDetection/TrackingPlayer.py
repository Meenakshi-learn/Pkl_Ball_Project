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
# GPU CHECK AND SETUP
# -----------------------------
print("="*60)
print("JETSON NANO GPU STATUS CHECK")
print("="*60)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    device = 'cuda:0'
else:
    print("⚠️  WARNING: CUDA not available, using CPU (will be slow!)")
    device = 'cpu'

print(f"Using device: {device}")
print("="*60 + "\n")

# -----------------------------
# LOAD YOLO MODELS WITH GPU
# -----------------------------
print("Loading YOLO models...")
model = YOLO("yolov8n.pt")
pose_model = YOLO("yolov8n-pose.pt")

# Explicitly set models to use GPU
if cuda_available:
    model.to(device)
    pose_model.to(device)
    print("✓ Models loaded on GPU")
else:
    print("✓ Models loaded on CPU")

# -----------------------------
# INPUT VIDEO (Jetson Nano)
# -----------------------------
video_path = "/home/univision/Niranjan/video1 (online-video-cutter.com).mp4"
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
output_video_path = "/home/univision/Niranjan/half_court_players_polygon_gpu.mp4"
out = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps, (W, H)
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
# TRACKING VARIABLES
# -----------------------------
stable_id_counter = 1
player_history = {}
DISTANCE_THRESHOLD = 200
FRAME_MEMORY = 300


# ---------------------------------------------------
# EXTRACT SIMPLE FEATURES FOR TRACKING
# ---------------------------------------------------
def get_bbox_features(x1, y1, x2, y2, frame):
    w = x2 - x1
    h = y2 - y1

    roi = frame[y1:y2, x1:x2]
    avg_color = roi.mean(axis=(0, 1)) if roi.size > 0 else np.array([0, 0, 0])

    return {
        "width": w,
        "height": h,
        "aspect_ratio": h / w if w > 0 else 0,
        "color": avg_color
    }


# ---------------------------------------------------
# FIXED: FIND EXTREME BODY EDGES USING SAFE CONTOURS
# ---------------------------------------------------
def find_contour_extremes(x1, y1, x2, y2, frame):
    roi = frame[y1:y2, x1:x2].copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    left_idx = largest_contour[:, :, 0].argmin()
    right_idx = largest_contour[:, :, 0].argmax()
    top_idx = largest_contour[:, :, 1].argmin()
    bottom_idx = largest_contour[:, :, 1].argmax()

    leftmost = tuple(largest_contour[left_idx][0])
    rightmost = tuple(largest_contour[right_idx][0])
    topmost = tuple(largest_contour[top_idx][0])
    bottommost = tuple(largest_contour[bottom_idx][0])

    return {
        "leftmost": (leftmost[0] + x1, leftmost[1] + y1),
        "rightmost": (rightmost[0] + x1, rightmost[1] + y1),
        "topmost": (topmost[0] + x1, topmost[1] + y1),
        "bottommost": (bottommost[0] + x1, bottommost[1] + y1),
    }


# ---------------------------------------------------
# EXTRACT BODY PART COORDINATES
# ---------------------------------------------------
def extract_body_part_coordinates(x1, y1, x2, y2, keypoints=None, frame=None):

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    w = x2 - x1
    h = y2 - y1

    parts = {
        "center_x": cx, "center_y": cy,
        "left_foot_x": None, "left_foot_y": None,
        "right_foot_x": None, "right_foot_y": None,
        "left_arm_x": None, "left_arm_y": None,
        "right_arm_x": None, "right_arm_y": None,
    }

    extremes = find_contour_extremes(x1, y1, x2, y2, frame)

    if keypoints is not None and len(keypoints) > 0:
        kpts = keypoints[0]

        if kpts[15][2] > 0.3:
            parts["left_foot_x"] = int(kpts[15][0])
            parts["left_foot_y"] = int(kpts[15][1])

        if kpts[16][2] > 0.3:
            parts["right_foot_x"] = int(kpts[16][0])
            parts["right_foot_y"] = int(kpts[16][1])

        if kpts[9][2] > 0.3:
            parts["left_arm_x"] = int(kpts[9][0])
            parts["left_arm_y"] = int(kpts[9][1])

        if kpts[10][2] > 0.3:
            parts["right_arm_x"] = int(kpts[10][0])
            parts["right_arm_y"] = int(kpts[10][1])

    # FALLBACKS
    if parts["left_foot_x"] is None:
        parts["left_foot_x"] = x1 + w // 4
        parts["left_foot_y"] = y2

    if parts["right_foot_x"] is None:
        parts["right_foot_x"] = x1 + 3 * w // 4
        parts["right_foot_y"] = y2

    if parts["left_arm_x"] is None:
        parts["left_arm_x"] = x1
        parts["left_arm_y"] = y1 + h // 3

    if parts["right_arm_x"] is None:
        parts["right_arm_x"] = x2
        parts["right_arm_y"] = y1 + h // 3

    return parts


# ---------------------------------------------------
# FIXED PLAYER MATCHING SYSTEM
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
        new_id = stable_id_counter
        stable_id_counter += 1
        player_history[new_id] = {
            "last_pos": (cx, cy),
            "last_frame": frame_no,
            "appearance": features,
        }
        return new_id

    best_match = None
    best_score = float("inf")

    for sid, info in player_history.items():
        px, py = info["last_pos"]

        d = math.hypot(cx - px, cy - py)
        color_d = np.linalg.norm(features["color"] - info["appearance"]["color"])
        size_d = abs(features["aspect_ratio"] - info["appearance"]["aspect_ratio"])
        score = d + (color_d * 2) + (size_d * 50)

        if score < best_score:
            best_score = score
            best_match = sid

    if best_match is None or best_score > DISTANCE_THRESHOLD * 3:
        new_id = stable_id_counter
        stable_id_counter += 1
        player_history[new_id] = {
            "last_pos": (cx, cy),
            "last_frame": frame_no,
            "appearance": features,
        }
        return new_id

    player_history[best_match]["last_pos"] = (cx, cy)
    player_history[best_match]["last_frame"] = frame_no

    return best_match


# ---------------------------------------------------
# MAIN VIDEO LOOP WITH GPU MONITORING
# ---------------------------------------------------
player_coordinates = []
frame_no = 0

print("▶ Processing video... Press Q to quit display.")
print("Monitoring GPU usage...\n")

import time
start_time = time.time()
last_gpu_check = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1

    # GPU-accelerated inference
    results = model.track(frame, conf=0.4, classes=[0], tracker="bytetrack.yaml", 
                         persist=True, device=device, verbose=False)
    pose_out = pose_model(frame, conf=0.4, device=device, verbose=False)

    kpts = None
    if hasattr(pose_out[0], "keypoints"):
        kpts = pose_out[0].keypoints.data.cpu().numpy()

    if results[0].boxes.id is not None:
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if half_court_polygon.contains(Point(cx, cy)):
                features = get_bbox_features(x1, y1, x2, y2, frame)
                sid = match_player(cx, cy, features, frame_no)

                parts = extract_body_part_coordinates(x1, y1, x2, y2, keypoints=kpts, frame=frame)

                player_coordinates.append([
                    frame_no, sid,
                    parts["center_x"], parts["center_y"],
                    parts["left_foot_x"], parts["left_foot_y"],
                    parts["right_foot_x"], parts["right_foot_y"],
                    parts["left_arm_x"], parts["left_arm_y"],
                    parts["right_arm_x"], parts["right_arm_y"],
                ])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"ID {sid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Show GPU usage every 5 seconds
    if cuda_available and time.time() - last_gpu_check > 5.0:
        gpu_mem = torch.cuda.memory_allocated(0) / 1e6
        gpu_mem_cached = torch.cuda.memory_reserved(0) / 1e6
        print(f"Frame {frame_no} | GPU Memory: {gpu_mem:.0f}MB (cached: {gpu_mem_cached:.0f}MB)")
        last_gpu_check = time.time()

    # Display processing info on frame
    elapsed = time.time() - start_time
    current_fps = frame_no / elapsed if elapsed > 0 else 0
    
    cv2.putText(frame, f"Frame: {frame_no} | FPS: {current_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Device: {device.upper()}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if cuda_available else (0, 0, 255), 2)

    # --------------------------------
    # DISPLAY ON JETSON NANO SCREEN
    # --------------------------------
    cv2.imshow("Processed Video (Press Q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("⏹ Display stopped by user.")
        break

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

# ---------------------------------------------------
# PERFORMANCE SUMMARY
# ---------------------------------------------------
total_time = time.time() - start_time
avg_fps = frame_no / total_time if total_time > 0 else 0

print("\n" + "="*60)
print("PROCESSING SUMMARY")
print("="*60)
print(f"Total Frames: {frame_no}")
print(f"Total Time: {total_time:.2f} seconds")
print(f"Average FPS: {avg_fps:.2f}")
print(f"Device Used: {device}")

if cuda_available:
    print(f"Final GPU Memory: {torch.cuda.memory_allocated(0) / 1e6:.0f}MB")
    print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated(0) / 1e6:.0f}MB")

print("="*60)
print("🎥 Output Video Saved:", output_video_path)
print("📊 Total Tracked Data Points:", len(player_coordinates))
print("✅ DONE!")
print("="*60)