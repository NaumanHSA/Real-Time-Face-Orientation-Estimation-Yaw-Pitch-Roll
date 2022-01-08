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


def draw_slider(image, x_min, x_max):
    x_c = (x_min + x_max) // 2
    rec_start = (x_min, 26)
    rec_end = (x_max, 40)
    cv2.rectangle(image, rec_start, rec_end, GREEN_COLOR, -1)
    cv2.line(image, (x_c, 20), (x_c, 44), BLUE_COLOR, 2)
    cv2.line(image, (x_min, 20), (x_min, 44), BLUE_COLOR, 2)
    cv2.line(image, (x_max, 20), (x_max, 44), BLUE_COLOR, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Yaw Index:', (x_min - 120, 40), font, 0.6, BLUE_COLOR, 2, cv2.LINE_AA)
    cv2.putText(image, '-1', (x_min - 20, 70), font, 0.75, BLUE_COLOR, 2, cv2.LINE_AA)
    cv2.putText(image, '0', (x_c - 8, 70), font, 0.75, BLUE_COLOR, 2, cv2.LINE_AA)
    cv2.putText(image, '1', (x_max - 5, 70), font, 0.75, BLUE_COLOR, 2, cv2.LINE_AA)


def draw_horizontal_face_line(image, points, offset=100, color=YELLOW_COLOR):

    points = sorted(points, key=lambda x: x[0])
    x = [pnt[0] for pnt in points]    
    y = [pnt[1] for pnt in points]    

    # find the coefficients for the best fit line on x and y
    m, b = np.polyfit(x, y, 1)

    # find the extream points and draw the line
    x1 = points[0][0] - offset
    x2 = points[-1][0] + offset
    y1, y2 = int(m * x1 + b), int(m * x2 + b)
    cv2.line(image, (x1, y1), (x2, y2), color, 2)


# function to get the sum of eucledian distances between all the points
def get_distance(points):
    points = sorted(points, key=lambda x: x[0])
    distance = 0
    for i in range(len(points) - 1):
        distance += math.sqrt(((points[i + 1][0] - points[i][0]) ** 2) + ((points[i + 1][1] - points[i][1]) ** 2))
    return distance


def detect(
    image: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    right_face_indices: list,
    left_face_indices: list,
    vis=True):

    image_rows, image_cols, _ = image.shape
    right_face_points = []
    left_face_points = []

    circle_radius = 2

    for idx, landmark in enumerate(landmark_list.landmark):

        landmark_px = normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)

        # only draw the landmarks specified in the filter_lansmarks
        if idx in right_face_indices and landmark_px:
            right_face_points.append(landmark_px)
        elif idx in left_face_indices and landmark_px:
            left_face_points.append(landmark_px)
        else:
            continue

    scaled_value = -1
    if len(right_face_indices) > 0 and len(left_face_points):

        # computer disntance between cheekbones and eyes
        d1 = get_distance(left_face_points)
        d2 = get_distance(right_face_points)
        d = d1 + d2

        x_min = int(image_cols * 0.55)
        x_max = int(image_cols * 0.95)

        # scale the distances to image height
        scale_factor = ((x_max - x_min)) / d if d != 0 else 1

        center_scaled = int(d1 * scale_factor) + x_min
        center_scaled = max(x_min, min(center_scaled, x_max))
        pointer = (center_scaled, 33)        
        scaled_value = round((2 * ((center_scaled - x_min) / (x_max - x_min))) - 1, 2)

        if vis:
            # draw the slider
            draw_slider(image, x_min, x_max)
            cv2.circle(image, pointer, 10, BLUE_COLOR, -1)
            draw_horizontal_face_line(image, [*right_face_points, *left_face_points], offset=100)
        
    return scaled_value
    # return True if scaled_value < offset or scaled_value > -offset else False
