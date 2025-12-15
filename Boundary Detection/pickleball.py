import cv2
import numpy as np
import json
import os

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = r"C:\Users\Niranjan\pickleball\homography44\output3"
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "output_images")
MASKED_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "masked_for_detection")
JSON_PATH = os.path.join(OUTPUT_DIR, "hull_config.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(MASKED_OUTPUT_DIR, exist_ok=True)


# ============================================================
# FIND LINE INTERSECTIONS
# ============================================================

def line_intersection(line1, line2):
    """Find intersection point of two lines"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
    
    return int(px), int(py)


def find_four_corners(vertical_lines, horizontal_lines, img_shape):
    """Find the 4 corner points by intersecting vertical and horizontal lines"""
    h, w = img_shape[:2]
    corners = []
    
    if len(vertical_lines) >= 2 and len(horizontal_lines) >= 2:
        # Get the two main vertical lines (left and right)
        left_line = vertical_lines[0]
        right_line = vertical_lines[1] if len(vertical_lines) > 1 else vertical_lines[0]
        
        # Get the two main horizontal lines (top and bottom)
        top_line = horizontal_lines[0]
        bottom_line = horizontal_lines[1] if len(horizontal_lines) > 1 else horizontal_lines[0]
        
        # Find 4 intersection points
        top_left = line_intersection(left_line, top_line)
        top_right = line_intersection(right_line, top_line)
        bottom_right = line_intersection(right_line, bottom_line)
        bottom_left = line_intersection(left_line, bottom_line)
        
        corners = [top_left, top_right, bottom_right, bottom_left]
        
        # Filter out None values and out-of-bounds points
        corners = [c for c in corners if c is not None 
                   and 0 <= c[0] < w and 0 <= c[1] < h]
    
    return corners


# ============================================================
# ORDER POINTS CLOCKWISE
# ============================================================

def order_points_clockwise(pts):
    """Order points in clockwise order: TL, TR, BR, BL"""
    pts = np.array(pts, dtype=np.float32)
    
    # Sort by y-coordinate
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    
    # Top two points
    top_pts = sorted_by_y[:2]
    top_pts = top_pts[np.argsort(top_pts[:, 0])]  # Sort by x
    
    # Bottom two points
    bottom_pts = sorted_by_y[2:]
    bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]  # Sort by x
    
    # Return as [TL, TR, BR, BL]
    return np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.float32)


# ============================================================
# OFFSET POLYGON
# ============================================================

def offset_polygon(points, offset=50):
    """Expand polygon outward by a fixed pixel amount"""
    pts = np.array(points, dtype=np.float32)
    center = np.mean(pts, axis=0)
    vecs = pts - center
    norms = np.linalg.norm(vecs, axis=1).reshape(-1, 1)
    norms[norms == 0] = 1
    unit = vecs / norms
    return (pts + unit * offset).astype(np.int32)


# ============================================================
# SAVE / LOAD CONFIG
# ============================================================

def save_hull_config(inner_hull, outer_hull, H_matrix, filepath=JSON_PATH):
    """Save detected hulls and homography matrix to JSON"""
    config = {
        "inner_hull": inner_hull.tolist(),
        "outer_hull": outer_hull.tolist(),
        "homography": H_matrix.tolist() if H_matrix is not None else None
    }
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[✓] JSON saved at: {filepath}")


def load_hull_config(filepath=JSON_PATH):
    """Load saved hull configuration"""
    if not os.path.exists(filepath):
        return None, None, None

    with open(filepath, "r") as f:
        data = json.load(f)
    
    inner = np.array(data["inner_hull"], dtype=np.int32)
    outer = np.array(data["outer_hull"], dtype=np.int32)
    H = np.array(data["homography"], dtype=np.float32) if data["homography"] else None

    return inner, outer, H


# ============================================================
# HOMOGRAPHY
# ============================================================

def warp_court_homography(img, inner_hull, out_w=600, out_h=1320):
    """Apply homography transformation to get top-down view"""
    src = inner_hull.astype(np.float32)

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src, dst)
    warped = cv2.warpPerspective(img, H, (out_w, out_h))
    return warped, H


# ============================================================
# MASK OUTSIDE OUTER HULL FOR DETECTION
# ============================================================

def create_detection_mask(img, outer_hull):
    """Create a mask that blacks out everything outside the outer hull"""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [outer_hull], 255)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return masked_img, mask


# ============================================================
# MAIN PIPELINE: DETECT 4 CORNERS OF PICKLEBALL COURT
# ============================================================

def detect_pickleball_court(image_path, use_saved=False, offset=50):
    """
    Detect exactly 4 corner points of pickleball court using:
    1. HSV color filtering
    2. Canny edge detection
    3. Hough line detection
    4. Line intersection to find corners
    5. Create inner hull (4 corners)
    6. Create outer hull (50px offset)
    """
    
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Cannot load:", image_path)
        return

    # --------------------------------------------
    # LOAD SAVED MODE
    # --------------------------------------------
    if use_saved:
        inner, outer, H = load_hull_config()
        if inner is not None:
            inner_overlay = img.copy()
            outer_overlay = img.copy()
            combined_overlay = img.copy()
            
            # Draw corners as circles
            for corner in inner:
                cv2.circle(inner_overlay, tuple(corner), 10, (255, 0, 255), -1)
                cv2.circle(combined_overlay, tuple(corner), 10, (255, 0, 255), -1)
            
            cv2.polylines(inner_overlay, [inner], True, (0, 255, 255), 4)
            cv2.polylines(outer_overlay, [outer], True, (0, 0, 255), 4)
            cv2.polylines(combined_overlay, [inner], True, (0, 255, 255), 4)
            cv2.polylines(combined_overlay, [outer], True, (0, 0, 255), 4)
            
            warped = cv2.warpPerspective(img, H, (600, 1320))

            cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, "saved_inner_hull.jpg"), inner_overlay)
            cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, "saved_outer_hull.jpg"), outer_overlay)
            cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, "saved_combined.jpg"), combined_overlay)
            cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, "saved_topdown.jpg"), warped)
            
            masked_img, detection_mask = create_detection_mask(img, outer)
            cv2.imwrite(os.path.join(MASKED_OUTPUT_DIR, "masked_for_detection.jpg"), masked_img)
            cv2.imwrite(os.path.join(MASKED_OUTPUT_DIR, "detection_mask.jpg"), detection_mask)

            print("[✓] Loaded saved configuration")
            return warped

    print("[...] Detecting 4 Corners of Pickleball Court")

    # ============================================================
    # STEP 1: HSV COLOR FILTERING
    # ============================================================
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Blue court surface
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # White lines
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    combined_mask = cv2.bitwise_or(blue_mask, white_mask)
    
    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, "01_mask.jpg"), combined_mask)

    # ============================================================
    # STEP 2: CANNY EDGE DETECTION
    # ============================================================
    
    edges = cv2.Canny(combined_mask, 50, 150, apertureSize=3)
    cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, "02_edges.jpg"), edges)

    # ============================================================
    # STEP 3: HOUGH LINE DETECTION
    # ============================================================
    
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi / 180,
        threshold=80, 
        minLineLength=150, 
        maxLineGap=50
    )
    
    if lines is None:
        print("❌ No lines detected")
        return

    h, w = img.shape[:2]
    vertical_lines = []
    horizontal_lines = []
    
    line_debug = img.copy()
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        length = np.hypot(x2 - x1, y2 - y1)
        
        # Classify lines
        if 70 < angle < 110:  # Vertical
            vertical_lines.append((x1, y1, x2, y2, length))
            cv2.line(line_debug, (x1, y1), (x2, y2), (255, 0, 0), 2)
        elif angle < 20 or angle > 160:  # Horizontal
            horizontal_lines.append((x1, y1, x2, y2, length))
            cv2.line(line_debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, "03_lines.jpg"), line_debug)
    
    # Sort by length
    vertical_lines.sort(key=lambda x: x[4], reverse=True)
    horizontal_lines.sort(key=lambda x: x[4], reverse=True)
    
    print(f"[INFO] Found {len(vertical_lines)} vertical, {len(horizontal_lines)} horizontal lines")

    # ============================================================
    # STEP 4: SEPARATE LEFT/RIGHT AND TOP/BOTTOM LINES
    # ============================================================
    
    # Get left and right vertical lines
    left_verticals = [l for l in vertical_lines if min(l[0], l[2]) < w // 2]
    right_verticals = [l for l in vertical_lines if max(l[0], l[2]) > w // 2]
    
    # Get top and bottom horizontal lines
    top_horizontals = [l for l in horizontal_lines if min(l[1], l[3]) < h // 2]
    bottom_horizontals = [l for l in horizontal_lines if max(l[1], l[3]) > h // 2]
    
    if not (left_verticals and right_verticals and top_horizontals and bottom_horizontals):
        print("❌ Could not find all 4 boundary lines")
        return
    
    # Get the best lines from each side
    left_line = left_verticals[0][:4]
    right_line = right_verticals[0][:4]
    top_line = top_horizontals[0][:4]
    bottom_line = bottom_horizontals[0][:4]

    # ============================================================
    # STEP 5: FIND 4 CORNER INTERSECTIONS
    # ============================================================
    
    top_left = line_intersection(left_line, top_line)
    top_right = line_intersection(right_line, top_line)
    bottom_right = line_intersection(right_line, bottom_line)
    bottom_left = line_intersection(left_line, bottom_line)
    
    corners = [top_left, top_right, bottom_right, bottom_left]
    
    # Validate corners
    corners = [c for c in corners if c is not None and 0 <= c[0] < w and 0 <= c[1] < h]
    
    if len(corners) != 4:
        print(f"❌ Could not find exactly 4 corners (found {len(corners)})")
        return
    
    print(f"[✓] Found 4 corners: {corners}")
    
    # Order corners properly
    inner_hull = order_points_clockwise(np.array(corners))
    inner_hull = inner_hull.astype(np.int32)
    
    # Draw corners
    corner_debug = img.copy()
    for i, corner in enumerate(inner_hull):
        cv2.circle(corner_debug, tuple(corner), 15, (0, 255, 255), -1)
        cv2.putText(corner_debug, str(i), tuple(corner), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, "04_corners.jpg"), corner_debug)

    # ============================================================
    # STEP 6: CREATE INNER HULL (EXACT 4 CORNERS)
    # ============================================================
    
    print("[✓] Inner hull (4 corners):", inner_hull)

    # ============================================================
    # STEP 7: CREATE OUTER HULL (50px OFFSET)
    # ============================================================
    
    outer_hull = offset_polygon(inner_hull, offset)
    print(f"[✓] Outer hull created with {offset}px offset")

    # ============================================================
    # STEP 8: VISUALIZATION
    # ============================================================
    
    inner_overlay = img.copy()
    outer_overlay = img.copy()
    combined_overlay = img.copy()
    
    # Draw corner points
    for corner in inner_hull:
        cv2.circle(inner_overlay, tuple(corner), 10, (255, 0, 255), -1)
        cv2.circle(combined_overlay, tuple(corner), 10, (255, 0, 255), -1)
    
    # Draw hulls
    cv2.polylines(inner_overlay, [inner_hull], True, (0, 255, 255), 4)
    cv2.polylines(outer_overlay, [outer_hull], True, (0, 0, 255), 4)
    cv2.polylines(combined_overlay, [inner_hull], True, (0, 255, 255), 4)
    cv2.polylines(combined_overlay, [outer_hull], True, (0, 0, 255), 4)
    
    cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, "inner_hull.jpg"), inner_overlay)
    cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, "outer_hull.jpg"), outer_overlay)
    cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, "combined_hull.jpg"), combined_overlay)

    # ============================================================
    # STEP 9: HOMOGRAPHY TRANSFORMATION
    # ============================================================
    
    warped, H = warp_court_homography(img, inner_hull)
    cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, "court_topdown.jpg"), warped)

    # ============================================================
    # STEP 10: CREATE DETECTION MASK
    # ============================================================
    
    masked_img, detection_mask = create_detection_mask(img, outer_hull)
    cv2.imwrite(os.path.join(MASKED_OUTPUT_DIR, "masked_for_detection.jpg"), masked_img)
    cv2.imwrite(os.path.join(MASKED_OUTPUT_DIR, "detection_mask.jpg"), detection_mask)

    # ============================================================
    # STEP 11: SAVE CONFIGURATION
    # ============================================================
    
    save_hull_config(inner_hull, outer_hull, H)

    print("[✓] Detection complete!")
    print(f"    - Inner hull: 4 corner points")
    print(f"    - Outer hull: 50px offset boundary")
    print(f"    - Images saved in: {OUTPUT_IMAGES_DIR}")
    
    return warped


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    image_path = r"C:\Users\Niranjan\Downloads\image (2).jpg"
    detect_pickleball_court(image_path, use_saved=False, offset=50)
  
   