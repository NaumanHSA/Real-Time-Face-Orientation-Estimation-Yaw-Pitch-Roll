import cv2
import os, sys
import math
import numpy as np
import statistics

from mediapipe.framework.formats import landmark_pb2
from utils.general_utils import normalized_to_pixel_coordinates

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
YELLOW_COLOR = (0, 255, 255)


# this method draws the slider for visualizing the angle of roll
def draw_slider(image, x_min, x_max):

    image_rows, image_cols, _ = image.shape

    # define the rectangle start end points
    x_c = (x_min + x_max) // 2
    rec_y1 = image_rows - 34
    rec_y2 = image_rows - 20
    rec_start = (x_min, rec_y1)
    rec_end = (x_max, rec_y2)

    # draw the slider rectangle and points
    cv2.rectangle(image, rec_start, rec_end, GREEN_COLOR, -1)
    cv2.line(image, (x_min, rec_y1 - 5), (x_min, rec_y2 + 5), BLUE_COLOR, 2)
    cv2.line(image, (x_c, rec_y1 - 5), (x_c, rec_y2 + 5), BLUE_COLOR, 2)
    cv2.line(image, (x_max, rec_y1 - 5), (x_max, rec_y2 + 5), BLUE_COLOR, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Roll Angle:', (x_min - 120, rec_y2 - 5), font, 0.6, BLUE_COLOR, 2, cv2.LINE_AA)
    cv2.putText(image, '-90', (x_min - 20, rec_y1 - 20), font, 0.75, BLUE_COLOR, 2, cv2.LINE_AA)
    cv2.putText(image, '0', (x_c - 8, rec_y1 - 20), font, 0.75, BLUE_COLOR, 2, cv2.LINE_AA)
    cv2.putText(image, '+90', (x_max - 5, rec_y1 - 20), font, 0.75, BLUE_COLOR, 2, cv2.LINE_AA)

def detect(
    image: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    roll_indices: list,
    vis=True):

    image_rows, image_cols, _ = image.shape
    x = []
    y = []
    circle_radius = 2
    for idx, landmark in enumerate(landmark_list.landmark):
        landmark_px = normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)

        # only draw the landmarks specified in the filter_lansmarks
        if idx in roll_indices and landmark_px:
            x.append(landmark_px[0])
            y.append(landmark_px[1])
        else:
            continue
        
        if vis:
            cv2.circle(image, landmark_px, circle_radius, RED_COLOR, -1)

    angle = 90    
    if len(x) > 0 and len(y) > 0:
        # find the coefficients for the best fit line on x and y
        m, b = np.polyfit(x, y, 1)
            
        # straight line equation:
        # y = mx + b  -->     x = (y - b) / x
        y1, y2 = max(y), min(y)
        x1, x2 = int((y1 - b) / m), int((y2 - b) / m)

        # compute the angle between the eye centroids
        dY = y2 - y1
        dX = x2 - x1
        angle = np.degrees(np.arctan2(dY, dX)) + 180

        x_min = int(image_cols * 0.55)
        x_max = int(image_cols * 0.95)

        # scale the distances to image height
        scale_factor = ((x_max - x_min)) / 180

        center_scaled = int(angle * scale_factor) + x_min
        center_scaled = max(x_min, min(center_scaled, x_max))
        pointer = (center_scaled, image_rows - 27)
        
        if vis:
            # draw the slider and draw the pointer
            draw_slider(image, x_min, x_max)
            cv2.circle(image, pointer, 10, BLUE_COLOR, -1)

    print(angle)
    return (int(angle) - 90)
