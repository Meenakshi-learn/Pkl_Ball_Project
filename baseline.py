import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. Load image
# -------------------------------------------------------
path = "../frames/frame_00010.jpg"
img = cv2.imread(path)
orig = img.copy()

# -------------------------------------------------------
# 2. Gray
# -------------------------------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# -------------------------------------------------------
# 3. ROI: bottom blue region
# -------------------------------------------------------
top    = int(h * 0.40)
bottom = int(h * 0.98)
left   = int(w * 0.10)
right  = int(w * 0.90)

roi = gray[top:bottom, left:right]

# -------------------------------------------------------
# 4. Blur
# -------------------------------------------------------
blur = cv2.GaussianBlur(roi, (5,5), 1)

# -------------------------------------------------------
# 5. Canny edges
# -------------------------------------------------------
edges = cv2.Canny(blur, 40, 150)

# -------------------------------------------------------
# 6. Hough line detection
# -------------------------------------------------------
lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi/180,
    threshold=60,
    minLineLength=120,
    maxLineGap=40
)

# -------------------------------------------------------
# 7. Extract angles of all candidate lines
# -------------------------------------------------------
candidates = []

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Shift to full coordinates
        X1, Y1 = x1 + left, y1 + top
        X2, Y2 = x2 + left, y2 + top

        # Angle of line
        angle = np.degrees(np.arctan2(Y2 - Y1, X2 - X1))
        angle = abs(angle)   # ignore sign

        length = np.hypot(X2 - X1, Y2 - Y1)

        # Save all lines (we will filter by parallelism later)
        candidates.append((angle, length, X1, Y1, X2, Y2))

# -------------------------------------------------------
# 8. Find two long lines with nearly same angle
# -------------------------------------------------------
# Sort by angle
candidates.sort(key=lambda x: x[0])

best_pair = None
best_score = -1

for i in range(len(candidates)):
    for j in range(i+1, len(candidates)):
        angle1, len1, x1a, y1a, x1b, y1b = candidates[i]
        angle2, len2, x2a, y2a, x2b, y2b = candidates[j]

        if abs(angle1 - angle2) < 6:   # parallel tolerance in degrees
            score = len1 + len2        # choose longest pair
            if score > best_score:
                best_score = score
                best_pair = [(x1a,y1a,x1b,y1b), (x2a,y2a,x2b,y2b)]

# -------------------------------------------------------
# 9. Draw the two long parallel lines
# -------------------------------------------------------
result = orig.copy()

if best_pair is not None:
    for (X1,Y1,X2,Y2) in best_pair:
        cv2.line(result, (X1,Y1), (X2,Y2), (0,255,0), 6)
else:
    print("No parallel line pair found.")

plt.figure(figsize=(10,6))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Detected Two Long Parallel Court Side Boundaries")
plt.axis("off")
plt.show()
