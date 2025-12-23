import cv2

def binarize_frame(
    frame,
    blur_ksize=11,
    threshold=180
):
    """
    Convert input BGR frame to binary image suitable for court-line detection.
    Designed for indoor courts.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, blur_ksize, 75, 75)

    # Fixed threshold works better than adaptive for indoor lighting
    _, binary = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    return binary