import math
import statistics
import cv2
import numpy as np


WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)


def drawline(img, pt1, pt2, color, thickness=1, style="dotted", gap=20):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
        p = (x, y)
        pts.append(p)

    if style == "dotted":
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def drawfacerect(img, pt1, pt2, color, thickness=1, style="dotted"):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style)


def draw_results(image, 
    pitch_index, yaw_index, roll_angle, 
    is_pitch, is_yaw, is_roll, 
    pitch_message, yaw_message, roll_message
    ):

    h, w, _ = image.shape
    if w <= 1080:
        font_scale = 0.55
        rec_size = 460
        diff = 35
    else:
        font_scale = 1
        diff = 60
        rec_size = 1000

    font = cv2.FONT_HERSHEY_SIMPLEX

    texts = [
    'Pitch Index: ' + str(pitch_index) + ' -> ' + pitch_message,
    'Yaw Index: ' + str(yaw_index) + ' -> ' + yaw_message,
    'Roll Angle: ' + str(roll_angle) + ' deg -> ' + roll_message,
    ]

    cv2.rectangle(image, (10, 10), (rec_size, (diff * len(texts)) + 20), (204, 153, 0), -1)
    pitch_color = (0, 0xFF, 0) if is_pitch else (0, 0, 0xFF)
    yaw_color = (0, 0xFF, 0) if is_yaw else (0, 0, 0xFF)
    roll_color = (0, 0xFF, 0) if is_roll else (0, 0, 0xFF)

    # insert information text to video frame
    cv2.putText(image, texts[0], (20, diff * 1), font, font_scale, pitch_color, 2, cv2.FONT_HERSHEY_SIMPLEX)
    cv2.putText(image, texts[1], (20, diff * 2), font, font_scale, yaw_color, 2, cv2.FONT_HERSHEY_SIMPLEX)    
    cv2.putText(image, texts[2], (20, diff * 3), font, font_scale, roll_color, 2, cv2.FONT_HERSHEY_COMPLEX_SMALL)
