import numpy as np
import cv2
import math


def normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int, image_height: int):
    """Converts normalized value pair to pixel coordinates."""
    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (
            value < 1 or math.isclose(1, value)
        )

    if not (
        is_valid_normalized_value(normalized_x)
        and is_valid_normalized_value(normalized_y)
    ):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = int(min(math.floor(normalized_x * image_width), image_width - 1))
    y_px = int(min(math.floor(normalized_y * image_height), image_height - 1))
    return x_px, y_px


def get_face_rect(image, landmark_list, offset=1):
    image_rows, image_cols, _ = image.shape
    xmin = min([landmark.x for landmark in landmark_list.landmark])
    ymin = min([landmark.y for landmark in landmark_list.landmark])
    xmax = max([landmark.x for landmark in landmark_list.landmark])
    ymax = max([landmark.y for landmark in landmark_list.landmark])

    xmin, ymin = normalized_to_pixel_coordinates(xmin, ymin, image_cols, image_rows)
    xmax, ymax = normalized_to_pixel_coordinates(xmax, ymax, image_cols, image_rows)

    # rescale the box offset times
    w = int(((xmax - xmin) // 2) * offset)
    h = int(((ymax - ymin) // 2) * offset)

    xmin -= w
    xmax += w
    ymin -= h
    # ymax += h

    return (xmin, ymin), (xmax, ymax)


def resize(image, dim=None, inter=cv2.INTER_AREA):
    # if dim is None, then return the original image
    if dim is None:
        return image

    width, height = dim
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # calculate the ratio of the width and construct the
    # dimensions
    r = width / float(w)
    dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    (h, w) = resized.shape[:2]
    resized = cv2.copyMakeBorder(resized, max(0, (height - h) // 2), max(0, (height - h) // 2), 0, 0, cv2.BORDER_CONSTANT)

    # return the resized image
    return resized