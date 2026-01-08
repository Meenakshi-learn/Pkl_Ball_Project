import numpy as np
from pathlib import Path
import cv2

from .world.court_model import half_court, center_line, nvz_line

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE = Path(__file__).parent
ROOT = BASE.parent
H_PATH = ROOT / "shared" / "H_frozen.npy"
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)

assert H_PATH.exists(), "❌ H_frozen.npy not found. Run Stage-1 first."

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def project(points, H):
    pts = np.array(points, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    H = np.load(H_PATH)

    anchored = {
        "court_polygon": project(half_court(), H),
        "center_line": project(center_line(), H),
        "nvz_line": project(nvz_line(), H),
    }

    np.save(OUT / "anchored_court.npy", anchored)
    print("✅ Stage-2: anchored_court.npy saved")

if __name__ == "__main__":
    main()