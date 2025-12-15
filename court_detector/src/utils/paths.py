import os

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input image path
INPUT_IMAGE = os.path.join(BASE_DIR, "input", "court.jpg")

# Output directories
STAGE_DIR = os.path.join(BASE_DIR, "output", "stages/")
FINAL_DIR = os.path.join(BASE_DIR, "output", "final/")

# Create dirs if missing
os.makedirs(STAGE_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)