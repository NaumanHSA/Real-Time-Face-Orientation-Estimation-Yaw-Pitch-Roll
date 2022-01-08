import cv2
import mediapipe as mp
import os, sys
import statistics
import math
from datetime import datetime

import utils.drawing_utils as drawing_utils
from utils.general_utils import get_face_rect, resize

import facemash_indices as consts
import CONFIG as cfg
from detectors import roll, pitch, yaw

mp_face_mesh = mp.solutions.face_mesh


# For webcam input:
cap = cv2.VideoCapture(0)

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
YELLOW_COLOR = (0, 255, 255)


with mp_face_mesh.FaceMesh(
    max_num_faces=cfg.MAX_NUMBER_FACES,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        
        pitch_index, yaw_index, roll_angle = 'n/a', 'n/a', 'n/a'
        is_pitch, is_yaw, is_roll = False, False, False
        pitch_message, yaw_message, roll_message, = '', '', ''
        try:
            # resize the image
            image = resize(image, dim=cfg.IMAGE_SIZE)
            image_backup = image.copy()     # create a copy image

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    
                    # draw face rectangle
                    start_pnt, end_pnt = get_face_rect(image, face_landmarks, offset=0.5)
                    drawing_utils.drawfacerect(image, start_pnt, end_pnt, GREEN_COLOR, thickness=2, style="dashed")
                    
                    # detect face rotation on z-axis (also called roll)
                    roll_angle = roll.detect(image, face_landmarks, 
                        consts.FACEMESH_ROLL_POINTS,
                        vis=cfg.VIS
                        )

                    if -cfg.ROLL_THRESHOLD <= roll_angle <= cfg.ROLL_THRESHOLD:
                        is_roll = True
                        roll_message = "Roll is Good"
                    elif roll_angle < -cfg.ROLL_THRESHOLD:
                        is_roll = False
                        roll_message = "Rotating Anticlockwise"
                    elif roll_angle > cfg.ROLL_THRESHOLD:
                        is_roll = False
                        roll_message = "Rotating Clockwise"

                    # detect face rotation on z-axis (also called roll)
                    pitch_index = pitch.detect(image, face_landmarks, 
                        consts.FACEMESH_EYES_CENTER,
                        consts.FACEMESH_CHAIN_CENTER,
                        consts.FACEMESH_NOSE_TIP,
                        vis=cfg.VIS
                        )

                    if -cfg.PITCH_THRESHOLD <= pitch_index <= cfg.PITCH_THRESHOLD:
                        is_pitch = True
                        pitch_message = "Pitch is Good"
                    elif pitch_index < -cfg.PITCH_THRESHOLD:
                        is_pitch = False
                        pitch_message = "Looking Downwards"
                    elif pitch_index > cfg.PITCH_THRESHOLD:
                        is_pitch = False
                        pitch_message = "Looking Upwards"

                    # detect face rotation on z-axis (also called roll)
                    yaw_index = yaw.detect(
                        image, 
                        face_landmarks, 
                        consts.FACEMESH_LEFT_POINTS,
                        consts.FACEMESH_RIGHT_POINTS,
                        vis=cfg.VIS
                        )

                    if -cfg.YAW_THRESHOLD <= yaw_index <= cfg.YAW_THRESHOLD:
                        is_yaw = True
                        yaw_message = "Yaw is Good"
                    elif yaw_index < -cfg.YAW_THRESHOLD:
                        is_yaw = False
                        yaw_message = "Looking towards Left"
                    elif yaw_index > cfg.YAW_THRESHOLD:
                        is_yaw = False
                        yaw_message = "Looking towards Right"

        except Exception as e:
            pass
        
        # draw the results
        drawing_utils.draw_results(
            image, 
            pitch_index, yaw_index, roll_angle, 
            is_pitch, is_yaw, is_roll,
            pitch_message, yaw_message, roll_message)

        # Flip the image horizontally for a selfie-view display.
        # cv2.imshow("MediaPipe Face Mesh", cv2.flip(image, 1))
        cv2.imshow("MediaPipe Face Mesh", image)

        key = cv2.waitKey(1)
        if key == 32 or key == ord('c'):
            # save the current image
            image_name = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + '.jpg'
            image_path = os.path.join(cfg.OUTPUT_IMAGES_PATH, image_name)
            cv2.imwrite(image_path, image_backup)

            # open the last captured image in a new window
            cv2.imshow("Last Captured Image", image_backup)
        elif key & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
