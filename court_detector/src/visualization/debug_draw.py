import cv2

def draw_lines(img, lines, color):
    out = img.copy()
    for (x1,y1,x2,y2) in lines:
        cv2.line(out, (x1,y1), (x2,y2), color, 3)
    return out


def draw_points(img, points):
    out = img.copy()
    for i,(x,y) in enumerate(points):
        cv2.circle(out,(x,y),6,(0,0,255),-1)
        cv2.putText(out,str(i+1),(x+5,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    return out
