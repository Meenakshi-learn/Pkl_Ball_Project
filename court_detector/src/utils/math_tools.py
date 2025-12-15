import numpy as np

def line_length(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def intersection(l1, l2):
    (x1,y1,x2,y2) = l1
    (x3,y3,x4,y4) = l2

    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0:
        return None

    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom

    return (int(px), int(py))