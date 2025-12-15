import numpy as np
from sklearn.linear_model import RANSACRegressor


# =========================================================
#   Robust RANSAC FITTING: Horizontal (y = m*x + b)
# =========================================================
def fit_ransac_line(points):
    """
    Fit a robust 2D line using RANSAC.
    Assumes: y = m*x + b  (mostly horizontal lines)
    Returns: (m, b)
    """
    if len(points) < 2:
        return None

    pts = np.array(points, dtype=float)
    X = pts[:, 0].reshape(-1, 1)   # x
    y = pts[:, 1]                  # y

    try:
        ransac = RANSACRegressor(
            min_samples=2,
            residual_threshold=2.5,
            max_trials=200
        )
        ransac.fit(X, y)
        m = float(ransac.estimator_.coef_[0])
        b = float(ransac.estimator_.intercept_)
        return m, b
    except Exception:
        return None


# =========================================================
#   Robust RANSAC FITTING: Vertical (x = m*y + b)
# =========================================================
def fit_ransac_vertical(points):
    """
    Fit robust vertical-ish line.
    Assumes: x = m*y + b
    """
    if len(points) < 2:
        return None

    pts = np.array(points, dtype=float)
    X = pts[:, 1].reshape(-1, 1)   # use y as input
    y = pts[:, 0]                  # output x

    try:
        ransac = RANSACRegressor(
            min_samples=2,
            residual_threshold=2.5,
            max_trials=200
        )
        ransac.fit(X, y)
        m = float(ransac.estimator_.coef_[0])
        b = float(ransac.estimator_.intercept_)
        return m, b
    except Exception:
        return None


# =========================================================
#   Convert a Hough segment → many samples for RANSAC
# =========================================================
def sample_segment(seg, num_points=40):
    x1, y1, x2, y2 = seg
    xs = np.linspace(x1, x2, num_points)
    ys = np.linspace(y1, y2, num_points)
    return list(zip(xs, ys))


def fit_ransac_from_segments(segments, vertical=False):
    """
    Fit RANSAC line from multiple Hough line segments.
    Useful if Hough returns jagged/broken lines.
    """
    pts = []
    for seg in segments:
        pts.extend(sample_segment(seg))

    if len(pts) < 2:
        return None

    if vertical:
        return fit_ransac_vertical(pts)
    else:
        return fit_ransac_line(pts)


# =========================================================
#   Convert final m,b line → full-screen endpoints
# =========================================================
def horizontal_line_to_endpoints(m, b, W):
    """
    Convert y = m*x + b into (x1,y1,x2,y2) spanning the image width.
    """
    y1 = int(b)
    y2 = int(m * W + b)
    return (0, y1, W, y2)


def vertical_line_to_endpoints(m, b, H):
    """
    Convert x = m*y + b into (x1,y1,x2,y2) spanning the image height.
    """
    x1 = int(b)
    x2 = int(m * H + b)
    return (x1, 0, x2, H)
