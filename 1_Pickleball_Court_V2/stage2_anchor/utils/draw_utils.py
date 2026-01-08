import cv2
import numpy as np

def draw_polygon(img, poly, color=(0,0,255), thickness=2):
    pts = np.array(poly, dtype=np.int32)
    cv2.polylines(
        img,
        [pts.reshape(-1,1,2)],
        True,
        color,
        thickness,
        cv2.LINE_AA
    )
def draw_infinite_line(img, line, color, thickness=1):
    vx, vy, x0, y0 = line
    h, w = img.shape[:2]

    p1 = (int(x0 - vx * 2000), int(y0 - vy * 2000))
    p2 = (int(x0 + vx * 2000), int(y0 + vy * 2000))

    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    
def draw_segment(img, p1, p2, color, thickness=2):
    p1 = tuple(map(int, p1))
    p2 = tuple(map(int, p2))
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    
def clamp_segment(p1, p2, w, margin=4):
    x1, y1 = p1
    x2, y2 = p2
    return (
        (max(margin, min(w-margin, x1)), y1),
        (max(margin, min(w-margin, x2)), y2)
    )
    
def extend_line_to_image(line, width):
    """
    Extend a line (vx, vy, x0, y0) to left/right image borders
    """
    vx, vy, x0, y0 = line

    # Avoid vertical instability
    if abs(vx) < 1e-6:
        return None

    # x = 0
    t0 = (0 - x0) / vx
    y0_ext = y0 + t0 * vy

    # x = width
    t1 = (width - x0) / vx
    y1_ext = y0 + t1 * vy

    return (0, y0_ext), (width, y1_ext)