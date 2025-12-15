# court_detector/src/io/config.py

import os

class Config:
    """Simple configuration handler."""

    def __init__(self, input_path=None, stage_dir="output/stages/", final_dir="output/final/"):
        base = os.path.dirname(os.path.abspath(__file__))

        # Root of your package
        self.project_root = os.path.abspath(os.path.join(base, "../.."))

        # Input image path
        if input_path is None:
            self.INPUT_IMAGE = os.path.join(self.project_root, "input", "court.jpg")
        else:
            self.INPUT_IMAGE = input_path

        # Stage output dir
        self.STAGE_DIR = os.path.join(self.project_root, stage_dir)
        os.makedirs(self.STAGE_DIR, exist_ok=True)

        # Final output dir
        self.FINAL_DIR = os.path.join(self.project_root, final_dir)
        os.makedirs(self.FINAL_DIR, exist_ok=True)
