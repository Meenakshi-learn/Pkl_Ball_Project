import json
import cv2

def save_image(path, img):
    cv2.imwrite(path, img)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
