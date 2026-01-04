# Half Court Detection – Pickleball

This repository implements a **deterministic, geometry-driven pipeline** for
near-side pickleball half-court detection and alignment from fixed or
broadcast-style camera views.

The system avoids machine learning and instead relies on **classical computer
vision, physical court constraints, and homography-based alignment** to achieve
stable and perspective-correct court overlays.

---

## Key Features

- Near-side ROI extraction for clutter reduction
- Hough-based line detection and classification
- Baseline-first geometric reasoning
- Robust sideline selection
- NVZ-constrained court reconstruction
- Homography-based perspective alignment
- Stable, frame-consistent court overlay

---

## Project Structure

