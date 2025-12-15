import cv2

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError("Cannot load image: " + path)
    return img
