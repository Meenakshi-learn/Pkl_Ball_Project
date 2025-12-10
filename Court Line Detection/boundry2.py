import cv2
import numpy as np
import json
import os

def line_length(line):
    """Calculate length of a line"""
    x1, y1, x2, y2 = line
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def extend_line(x1, y1, x2, y2, length=2000):
    """Extend a line in both directions"""
    dx = x2 - x1
    dy = y2 - y1
    norm = np.sqrt(dx**2 + dy**2)
    
    if norm == 0:
        return x1, y1, x2, y2
    
    dx /= norm
    dy /= norm
    
    new_x1 = int(x1 - dx * length)
    new_y1 = int(y1 - dy * length)
    new_x2 = int(x2 + dx * length)
    new_y2 = int(y2 + dy * length)
    
    return new_x1, new_y1, new_x2, new_y2

def line_intersection(line1, line2):
    """Find intersection point of two lines"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-6:
        return None
    
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    
    return int(px), int(py)

def save_boundary_config(boundary_box, filepath=r"C:\boundry1\boundary_config.json"):
    """Save boundary configuration to file"""
    config = {
        "boundary_box": boundary_box
    }
    with open(filepath, 'w') as f:
        json.dump(config, f)
    print(f"Boundary configuration saved to: {filepath}")

def load_boundary_config(filepath=r"C:\boundry1\boundary_config.json"):
    """Load boundary configuration from file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            config = json.load(f)
        return config["boundary_box"]
    return None

def detect_court_boundary(image_path, use_saved=False):
    """Detect court edges and draw fixed outer boundary with 50px offset"""
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image")
        return
    
    h, w = img.shape[:2]
    output = img.copy()
    
    # Try to load saved boundary configuration
    if use_saved:
        boundary_box = load_boundary_config()
        if boundary_box is not None:
            print("Using saved fixed boundary configuration")
            min_x, min_y, max_x, max_y = boundary_box
            
            # Draw the fixed outer boundary (red rectangle)
            cv2.rectangle(output, (min_x, min_y), (max_x, max_y), (0, 0, 255), 4)
            
            cv2.imshow('Fixed Court Boundary', output)
            cv2.imwrite(r"C:\boundry1\court_boundary.jpg", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return output
    
    # Detect court boundary for the first time
    print("Detecting court edge points from longest parallel lines...")
    
    # Convert to grayscale and detect white lines
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Detect white lines
    lower_white = np.array([0, 0, 170])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_white = cv2.dilate(mask_white, kernel, iterations=1)
    
    # Edge detection
    edges = cv2.Canny(mask_white, 50, 150, apertureSize=3)
    
    # Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=60, maxLineGap=30)
    
    if lines is None:
        print("No lines detected")
        return
    
    # Classify lines by orientation
    vertical_lines = []
    horizontal_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        length = line_length((x1, y1, x2, y2))
        
        if 60 < angle < 120:  # Vertical (parallel sides)
            vertical_lines.append((x1, y1, x2, y2, length))
        elif angle < 30 or angle > 150:  # Horizontal (top and bottom)
            horizontal_lines.append((x1, y1, x2, y2, length))
    
    print(f"Detected {len(vertical_lines)} vertical and {len(horizontal_lines)} horizontal lines")
    
    if len(vertical_lines) < 2 or len(horizontal_lines) < 2:
        print("Not enough lines detected")
        return
    
    # Sort by length and get longest lines
    vertical_lines.sort(key=lambda x: x[4], reverse=True)
    horizontal_lines.sort(key=lambda x: x[4], reverse=True)
    
    # Find longest left and right parallel lines
    left_candidates = [l for l in vertical_lines if min(l[0], l[2]) < w//2]
    right_candidates = [l for l in vertical_lines if max(l[0], l[2]) > w//2]
    
    if not left_candidates or not right_candidates:
        left_line = min(vertical_lines, key=lambda l: min(l[0], l[2]))[:4]
        right_line = max(vertical_lines, key=lambda l: max(l[0], l[2]))[:4]
    else:
        left_line = left_candidates[0][:4]
        right_line = right_candidates[0][:4]
    
    # Find longest top and bottom lines
    top_candidates = [l for l in horizontal_lines if min(l[1], l[3]) < h//2]
    bottom_candidates = [l for l in horizontal_lines if max(l[1], l[3]) > h//2]
    
    if not top_candidates or not bottom_candidates:
        top_line = min(horizontal_lines, key=lambda l: min(l[1], l[3]))[:4]
        bottom_line = max(horizontal_lines, key=lambda l: max(l[1], l[3]))[:4]
    else:
        top_line = top_candidates[0][:4]
        bottom_line = bottom_candidates[0][:4]
    
    print(f"Longest left line: length={line_length(left_line):.0f}px")
    print(f"Longest right line: length={line_length(right_line):.0f}px")
    print(f"Longest top line: length={line_length(top_line):.0f}px")
    print(f"Longest bottom line: length={line_length(bottom_line):.0f}px")
    
    # Extend lines for intersection
    left_ext = extend_line(*left_line)
    right_ext = extend_line(*right_line)
    top_ext = extend_line(*top_line)
    bottom_ext = extend_line(*bottom_line)
    
    # Find 4 edge corner points
    tl = line_intersection(left_ext, top_ext)
    tr = line_intersection(right_ext, top_ext)
    br = line_intersection(right_ext, bottom_ext)
    bl = line_intersection(left_ext, bottom_ext)
    
    corners = [tl, tr, br, bl]
    
    # Check if all corners are valid
    if None in corners:
        print("Could not find all 4 edge corner points")
        return
    
    # Ensure corners are within reasonable bounds
    corners = [(max(0, min(w, x)), max(0, min(h, y))) for x, y in corners]
    
    print(f"\nDetected 4 edge corner points:")
    for i, corner in enumerate(corners):
        print(f"  Corner {i+1}: {corner}")
    
    # Calculate outer boundary with 50px offset
    offset = 50
    
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    
    min_x = max(0, min(xs) - offset)
    max_x = min(w, max(xs) + offset)
    min_y = max(0, min(ys) - offset)
    max_y = min(h, max(ys) + offset)
    
    # Save the fixed boundary configuration
    boundary_box = [min_x, min_y, max_x, max_y]
    save_boundary_config(boundary_box)
    
    # Draw ONLY the outer boundary (red rectangle with 50px offset)
    cv2.rectangle(output, (min_x, min_y), (max_x, max_y), (0, 0, 255), 4)
    
    print(f"\nOuter boundary (50px offset): ({min_x}, {min_y}) to ({max_x}, {max_y})")
    print("This boundary is now FIXED and will not change even if players interrupt")
    
    # Display results
    cv2.imshow('White Lines', mask_white)
    cv2.imshow('Fixed Court Boundary', output)
    
    # Save output
    output_path = r"C:\boundry1\court_boundary1.jpg"
    cv2.imwrite(output_path, output)
    print(f"\nSaved output to: {output_path}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return output

if __name__ == "__main__":
    image_path = r"C:\boundry1\Tennis 1.jpg"
    
    # First time: detect edge points and create fixed boundary
    print("=== First time detection - Creating fixed outer boundary ===")
    detect_court_boundary(image_path, use_saved=False)
    
    # For subsequent frames: use saved fixed boundary (players won't affect it)
    # Uncomment the line below to use the fixed boundary on other frames
    # detect_court_boundary(image_path, use_saved=True)