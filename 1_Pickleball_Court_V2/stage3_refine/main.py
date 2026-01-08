import cv2
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE = Path(__file__).parent
ROOT = BASE.parent

VIDEO_PATH = ROOT / "stage1_homography" / "input" / "pickleball_court.mp4"
ANCHOR_PATH = ROOT / "stage2_anchor" / "output" / "anchored_court.npy"
FINAL = BASE / "final"
FINAL.mkdir(exist_ok=True)

assert VIDEO_PATH.exists(), "❌ Video not found"
assert ANCHOR_PATH.exists(), "❌ anchored_court.npy not found"

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def snap_points_to_edges(points, edges, radius=6):
    h, w = edges.shape
    snapped = []

    for x, y in points:
        x, y = int(x), int(y)
        best, best_dist = None, radius * radius + 1

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                xx, yy = x + dx, y + dy
                if 0 <= xx < w and 0 <= yy < h and edges[yy, xx] > 0:
                    d = dx * dx + dy * dy
                    if d < best_dist:
                        best_dist = d
                        best = (xx, yy)

        if best is not None:
            snapped.append(best)

    return np.array(snapped, dtype=np.float32)


def refine_line(p0, p1, edges, samples=80):
    t = np.linspace(0, 1, samples)
    sampled = np.column_stack([
        p0[0] + t * (p1[0] - p0[0]),
        p0[1] + t * (p1[1] - p0[1])
    ])

    snapped = snap_points_to_edges(sampled, edges)
    if len(snapped) < 15:
        return p0, p1

    vx, vy, x0, y0 = cv2.fitLine(
        snapped, cv2.DIST_L2, 0, 0.01, 0.01
    )

    length = np.linalg.norm(p1 - p0)
    dx, dy = vx[0] * length / 2, vy[0] * length / 2
    center = np.array([x0[0], y0[0]])

    return center - [dx, dy], center + [dx, dy]

def fit_and_clip_horizontal(snapped_points, width):
    """
    Fit a strong horizontal line and clip it to image width
    """
    vx, vy, x0, y0 = cv2.fitLine(
        snapped_points, cv2.DIST_L2, 0, 0.01, 0.01
    )

    y = float(y0[0])
    return np.array([0, y]), np.array([width - 1, y])


def intersect_lines(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom

    return np.array([px, py], dtype=np.float32)


def assemble_polygon(geom):
    return geom["court_polygon"].astype(np.int32)

def clip_line_to_sides(line, left_side, right_side):
    p0, p1 = line
    L0, L1 = left_side
    R0, R1 = right_side

    i0 = intersect_lines(p0, p1, L0, L1)
    i1 = intersect_lines(p0, p1, R0, R1)

    if i0 is None or i1 is None:
        return np.array(line)

    return np.array([i0, i1], dtype=np.float32)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():

    anchored = np.load(ANCHOR_PATH, allow_pickle=True).item()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    assert cap.isOpened()

    ret, frame0 = cap.read()
    assert ret

    h0 = frame0.shape[0]
    roi_y = int(0.35 * h0)
    roi_h = h0 - roi_y
    roi_w = frame0.shape[1]

    out = cv2.VideoWriter(
        str(FINAL / "stage3_near_court_locked.avi"),
        cv2.VideoWriter_fourcc(*"MJPG"),
        25,
        (roi_w, roi_h),
        True
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    locked_geometry = None
    geometry_locked = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[roi_y:]
        vis = roi.copy()

        # ================= LOCK GEOMETRY ONCE =================
        if not geometry_locked:

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)

            locked_geometry = anchored.copy()
            cp = locked_geometry["court_polygon"]

            # ---------- STRONG BASELINE FIX ----------
            t = np.linspace(0, 1, 120)
            sampled = np.column_stack([
                cp[0][0] + t * (cp[1][0] - cp[0][0]),
                cp[0][1] + t * (cp[1][1] - cp[0][1])
            ])

            snapped = snap_points_to_edges(sampled, edges, radius=7)

            if len(snapped) > 25:
                base0, base1 = fit_and_clip_horizontal(snapped, roi_w)
            else:
                base0, base1 = cp[0], cp[1]

            # ---------- REFINE OTHER SIDES ----------
            
            left0, left1   = refine_line(cp[0], cp[3], edges)
            right0, right1 = refine_line(cp[1], cp[2], edges)

             #---------- REBUILD CORNERS (GEOMETRY-CLEAN) ----------

            BL = intersect_lines(base0, base1, left0, left1)
            BR = intersect_lines(base0, base1, right0, right1)
            if BL is None or BR is None:
                continue

            # FORCE BASELINE STRAIGHT
            y_base = 0.5 * (BL[1] + BR[1])
            BL = np.array([BL[0], y_base], dtype=np.float32)
            BR = np.array([BR[0], y_base], dtype=np.float32)

            # TOP EDGE FROM SIDELINES
            TL = intersect_lines(left0, left1, cp[2], cp[3])
            TR = intersect_lines(right0, right1, cp[2], cp[3])
            if TL is None or TR is None:
                continue

            # FORCE NVZ PERFECTLY STRAIGHT
            y_nvz = 0.5 * (TL[1] + TR[1])
            TL = np.array([TL[0], y_nvz], dtype=np.float32)
            TR = np.array([TR[0], y_nvz], dtype=np.float32)
            
            if all(c is not None for c in (BL, BR, TR, TL)):

                locked_geometry["court_polygon"] = np.array(
                    [BL, BR, TR, TL], dtype=np.float32
                )
                
                # NVZ is the TOP EDGE by definition
                locked_geometry["nvz_line"] = np.array([TL, TR], dtype=np.float32)

                # ---------------- INTERNAL LINES (NO EDGE SNAP) ----------------
                left_side  = np.array([BL, TL], dtype=np.float32)
                right_side = np.array([BR, TR], dtype=np.float32)

                # Center line: strong Stage-2 homography, clipped to refined court
                locked_geometry["center_line"] = clip_line_to_sides(
                    anchored["center_line"],
                    left_side,
                    right_side
                )

                # NVZ line: strong Stage-2 homography, clipped to refined court
                
                geometry_locked = True
                print("✅ Court geometry locked")
                
                print("BL:", BL)
                print("BR:", BR)
                print("TL:", TL)
                print("TR:", TR)
                # ---------- SAVE LOCKED GEOMETRY ONCE ----------
                STAGE3_OUT = ROOT / "stage3_refine" / "output"
                STAGE3_OUT.mkdir(exist_ok=True)

                np.save(
                    STAGE3_OUT / "locked_court_geometry.npy",
                    locked_geometry,
                    allow_pickle=True
                )

                print("💾 Stage-3 geometry saved")

     # ================= DRAW ONLY =================

        # Draw half-court boundary (ONCE)
        cv2.polylines(
            vis,
            [assemble_polygon(locked_geometry)],
            True,
            (0, 0, 255),
            3
        )
        
        # ---------- DRAW HALF-COURT CORNER POINTS ----------
        # Order: BL, BR, TR, TL
        corners = locked_geometry["court_polygon"]

        for (x, y) in corners:
            cv2.drawMarker(
                vis,
                (int(x), int(y)),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=8,
                thickness=2
            )

        out.write(vis)

    cap.release()
    out.release()
    print("✅ Stage-3 complete: post-homography optimization done")

if __name__ == "__main__":
    main()