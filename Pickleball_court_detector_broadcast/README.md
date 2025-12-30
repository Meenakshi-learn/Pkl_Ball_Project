# Pickleball Court Detector (Near-Side)

This module implements a real-time, near-side pickleball court detection pipeline from video input.
The objective is to reliably detect and track the half-court trapezoid, baseline, Non-Volley Zone (NVZ) line, and center line, while suppressing noisy surroundings.

The pipeline is designed as a deterministic, geometry-driven POC suitable for production constraints, not a long-term research prototype.

**Key Objectives**

Detect near-side half court only
Robust to lighting variations and indoor noise
Stable court geometry across video frames
Real-time capable (CPU-friendly)
Clear visual output (video)

**Input & Output**
**Input**

 Video file (.mp4)
Fixed camera, near-baseline perspective
Indoor pickleball court

**Output**
 Annotated video (near_side_court_output.mp4)

**Overlays:**

Near-side trapezoid (green)
NVZ line (yellow)
Center line (blue)

**Pipeline Architecture**
1️⃣ **Lighting Normalization**

Reduces illumination gradients

Improves edge consistency across frames

2️⃣ Region of Interest (ROI)

Bottom ~65% of frame

Removes audience, banners, ceiling, and far court

3️⃣ Dynamic Preprocessing

Adaptive filtering

Edge extraction optimized for court lines

4️⃣ Line Detection

Probabilistic Hough Transform

Line classification into:

Horizontal (baseline candidates)

Oblique (sidelines)

5️⃣ **Geometry-Only Trapezoid Estimation**

**Near-side trapezoid computed using:**
Baseline
Two sidelines
No learning, no priors — pure geometry

6️⃣ **Temporal Stabilization**

Sliding window median filter
Suppresses jitter and missed detections
Ensures trapezoid continuity across frames

7️⃣ **NVZ Line Derivation**

Derived after trapezoid stabilization
Uses real-world court ratios
Prevents NVZ drift and compression

8️⃣ **Masking & Refinement**

Everything outside trapezoid is suppressed
Improves NVZ and center line clarity

9️⃣ **Center Line Detection**

Computed geometrically between baseline and NVZ
Does not rely on extra Hough passes

🔟 **Video Output**

Clean annotated video
Frame-by-frame stable overlays

