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

CONNECTIONS_INDICES = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 17, 18, 200, 199, 175, 152]


def draw_slider(image, y_min, y_max):
    image_rows, image_cols, _ = image.shape
    y_c = (y_min + y_max) // 2

    rec_start = (image_cols - 30, y_min)
    rec_end = (image_cols - 20, y_max)
    cv2.rectangle(image, rec_start, rec_end, GREEN_COLOR, -1)
    cv2.line(image, (image_cols - 35, y_min), (image_cols - 15, y_min), BLUE_COLOR, 2)
    cv2.line(image, (image_cols - 35, y_c), (image_cols - 15, y_c), BLUE_COLOR, 2)
    cv2.line(image, (image_cols - 35, y_max), (image_cols - 15, y_max), BLUE_COLOR, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Pitch Index:', (image_cols - 120, y_min - 40), font, 0.6, BLUE_COLOR, 2, cv2.LINE_AA)

    cv2.putText(image, '1', (image_cols - 70, y_min), font, 0.75, BLUE_COLOR, 2, cv2.LINE_AA)
    cv2.putText(image, '0', (image_cols - 70, y_c), font, 0.75, BLUE_COLOR, 2, cv2.LINE_AA)
    cv2.putText(image, '-1', (image_cols - 70, y_max), font, 0.75, BLUE_COLOR, 2, cv2.LINE_AA)

def draw_connections(image, connections, color=YELLOW_COLOR):
    for i in range(len(CONNECTIONS_INDICES) - 1):
        cv2.line(image, connections[CONNECTIONS_INDICES[i]], connections[CONNECTIONS_INDICES[i + 1] ], color, 2)


def detect(
    image: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    eyes_center_indices: list,
    chain_indices: list,
    nose_indices: list,
    vis=True):

    image_rows, image_cols, _ = image.shape
    eyes_center_points, chain_points, nose_points = [], [], []
    connections = dict()
    for idx, landmark in enumerate(landmark_list.landmark):

        landmark_px = normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
        if idx in CONNECTIONS_INDICES and landmark_px:
            connections[idx] = landmark_px

        # only draw the landmarks specified in the filter_lansmarks
        if idx in eyes_center_indices and landmark_px:
            eyes_center_points.append(landmark_px)
        elif idx in chain_indices and landmark_px:
            chain_points.append(landmark_px)
        elif idx in nose_indices and landmark_px:
            nose_points.append(landmark_px)
        else:
            continue
        # cv2.circle(image, landmark_px, circle_radius, RED_COLOR, -1)

    # find the centers
    eyes_center = [int(statistics.mean(i)) for i in zip(*eyes_center_points)]
    chain_center = [int(statistics.mean(i)) for i in zip(*chain_points)]
    nose_center = [int(statistics.mean(i)) for i in zip(*nose_points)]

    scaled_value = 0
    if len(eyes_center) > 0 and len(nose_center) > 0 and len(chain_center) > 0:

        # computer disntance between nose and eyes_center, nose and chain
        d1 = (nose_center[1] - eyes_center[1])
        d2 = (chain_center[1] - nose_center[1])

        y_min = int(image_rows * 0.35)
        y_max = int(image_rows * 0.75)
        
        # scale the distances to image height
        scale_factor = ((y_max - y_min)) / (d1 + d2)
        
        nose_center_scaled = int(   d1 * scale_factor) + y_min
        pointer = (image_cols - 25, nose_center_scaled)

        text_origin_pointer = (image_cols - 120, nose_center_scaled - 5)
        scaled_value = round((2 * ((nose_center_scaled - y_max) / (y_min - y_max))) - 1, 2)

        if vis:
            # draw the slider
            draw_slider(image, y_min, y_max)
            cv2.circle(image, pointer, 10, BLUE_COLOR, -1)

            # draw the verticle face connections for visualizations
            draw_connections(image, connections)

    return scaled_value
