# Pickleball Court Detector (Near-Side)

This module focuses on detecting and stabilizing the **near-side court geometry**
under challenging lighting and perspective conditions.

### Pipeline
- Lighting normalization
- Dynamic + bilateral filtering
- Edge reinforcement
- Hough line detection
- Near-side trapezoid construction
- Homography-based stabilization

### Status
Near-side court detection implemented using OpenCV-based geometry.
Far-side detection under investigation.
