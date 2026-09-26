import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import os


# ==========================================
# MODEL PATH
# ==========================================

model_path = os.path.join(
    os.path.dirname(__file__),
    "face_landmarker.task"
)


# ==========================================
# CREATE FACE LANDMARKER
# ==========================================

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(
    options
)


# ==========================================
# FRAME TIMESTAMP
# ==========================================

frame_timestamp = 0


# ==========================================
# HEAD POSE FUNCTION
# ==========================================

def detect_head_pose(frame):

    global frame_timestamp

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect face landmarks
    results = detector.detect_for_video(
        mp_image,
        frame_timestamp
    )

    frame_timestamp += 1


    # ======================================
    # NO FACE
    # ======================================

    if not results.face_landmarks:
        return "NO FACE"


    # ======================================
    # GET FACE LANDMARKS
    # ======================================

    face_landmarks = results.face_landmarks[0]


    # Nose landmark
    nose = face_landmarks[1]

    x = nose.x
    y = nose.y


    # ======================================
    # HEAD DIRECTION
    # ======================================

    if x < 0.40:

        direction = "LOOKING LEFT"

    elif x > 0.60:

        direction = "LOOKING RIGHT"

    elif y < 0.35:

        direction = "LOOKING UP"

    elif y > 0.65:

        direction = "LOOKING DOWN"

    else:

        direction = "CENTER"


    return direction