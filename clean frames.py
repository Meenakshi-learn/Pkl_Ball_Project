import cv2
import numpy as np
import os

def remove_noise_keep_court_lines(image_path):
    """
    AGGRESSIVE filtering: Remove ALL noise (scoreboard, audience, UI).
    Keep ONLY actual court lines.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot read file: {image_path}")
        return None
    
    img_height, img_width = img.shape[:2]
    
    # Convert to HSV for green detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for bright green color
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    
    # Green mask
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # ========================================
    # STEP 1: AGGRESSIVE REGION MASKING
    # ========================================
    
    # Create a blank mask
    region_mask = np.zeros_like(green_mask)
    
    # Define ONLY the court playing area (trapezoid shape)
    # Adjust these percentages based on your camera angle
    court_region = np.array([
        [int(img_width * 0.15), int(img_height * 0.82)],  # Bottom left
        [int(img_width * 0.85), int(img_height * 0.82)],  # Bottom right
        [int(img_width * 0.92), int(img_height * 0.35)],  # Top right (far)
        [int(img_width * 0.08), int(img_height * 0.35)]   # Top left (far)
    ], dtype=np.int32)
    
    # Fill only this court region
    cv2.fillPoly(region_mask, [court_region], 255)
    
    # Apply strict region mask - EVERYTHING outside is removed
    green_mask = cv2.bitwise_and(green_mask, region_mask)
    
    # ========================================
    # STEP 2: REMOVE SMALL NOISE AGGRESSIVELY
    # ========================================
    
    # Strong morphological opening to remove all small objects
    kernel_open = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel_open, iterations=3)
    
    # ========================================
    # STEP 3: FILTER BY LINE PROPERTIES
    # ========================================
    
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_mask = np.zeros_like(green_mask)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Skip tiny contours immediately
        if area < 500:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate line properties
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        relative_area = area / (img_width * img_height)
        
        # Solidity check (how "solid" vs "scattered" the shape is)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6)
        
        # VERY STRICT FILTERING - Only accept clear lines
        is_line_like = (
            aspect_ratio > 5.0 and           # Very elongated
            relative_area > 0.0008 and       # Not too small
            relative_area < 0.08 and         # Not too large
            solidity > 0.7 and               # Solid shape (not scattered)
            min(w, h) < max(w, h) * 0.15     # Thin compared to length
        )
        
        if is_line_like:
            cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
    
    # ========================================
    # STEP 4: HOUGH LINE DETECTION (STRICT)
    # ========================================
    
    # Detect only strong, long straight lines
    lines = cv2.HoughLinesP(
        filtered_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=150,  # Very long lines only
        maxLineGap=25
    )
    
    final_mask = np.zeros_like(green_mask)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line properties
            length = np.hypot(x2 - x1, y2 - y1)
            
            # Only very long lines (court lines are long)
            if length < 180:
                continue
            
            # Calculate angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # STRICT angle filtering - only horizontal or vertical
            is_horizontal = (angle < 12) or (angle > 168)
            is_vertical = (78 < angle < 102)
            
            if not (is_horizontal or is_vertical):
                continue
            
            # Check if line is in valid court area
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Must be within court region
            if (mid_x < img_width * 0.10 or mid_x > img_width * 0.90 or
                mid_y < img_height * 0.30 or mid_y > img_height * 0.85):
                continue
            
            # Draw the valid line
            cv2.line(final_mask, (x1, y1), (x2, y2), 255, 3)
    
    # ========================================
    # STEP 5: FINAL CLEANUP
    # ========================================
    
    # Remove any isolated pixels
    kernel_final = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_final, iterations=2)
    
    # Apply region mask one final time to be absolutely sure
    final_mask = cv2.bitwise_and(final_mask, region_mask)
    
    # ========================================
    # STEP 6: DRAW ON ORIGINAL IMAGE
    # ========================================
    
    # Create result
    cleaned = img.copy()
    cleaned[final_mask > 0] = [0, 255, 0]
    
    return cleaned


# ===================================================
# ⭐ BATCH MODE: READ ALL IMAGES FROM INPUT FOLDER
# ===================================================
input_folder  = r"C:\boundry\framesoutput"
output_folder = r"C:\boundry\modified"

os.makedirs(output_folder, exist_ok=True)

count = 0
processed = 0
failed = 0

print("🎬 Starting AGGRESSIVE noise removal...\n")
print("⚡ Removing: Scoreboard, Audience, UI, All Noise")
print("✅ Keeping: Court Lines ONLY\n")

for file in sorted(os.listdir(input_folder)):
    # Process only images
    if not (file.lower().endswith(".jpg") or 
            file.lower().endswith(".png") or
            file.lower().endswith(".jpeg")):
        continue
    
    img_path = os.path.join(input_folder, file)
    cleaned_img = remove_noise_keep_court_lines(img_path)
    
    if cleaned_img is None:
        failed += 1
        continue
    
    save_path = os.path.join(output_folder, f"clean_{count:05d}.jpg")
    cv2.imwrite(save_path, cleaned_img)
    print(f"✔ [{count+1}] Cleaned: {file}")
    
    count += 1
    processed += 1

print("\n" + "="*60)
print(f"🎉 AGGRESSIVE CLEANING COMPLETE!")
print(f"✅ Successfully processed: {processed} frames")
if failed > 0:
    print(f"❌ Failed: {failed} frames")
print(f"📁 Output: {output_folder}")
print("="*60)