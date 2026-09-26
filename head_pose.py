import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# Create Face Landmarker
# -----------------------------

base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)


# -----------------------------
# Start Webcam
# -----------------------------

cap = cv2.VideoCapture(0)

frame_timestamp = 0

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read webcam")
        break

    # OpenCV: BGR
    # MediaPipe: RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert OpenCV image to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect face
    results = detector.detect_for_video(
        mp_image,
        frame_timestamp
    )

    frame_timestamp += 1

    # -----------------------------
    # Check face
    # -----------------------------

    if results.face_landmarks:

        face_landmarks = results.face_landmarks[0]

        # Nose landmark
        nose = face_landmarks[1]

        x = nose.x
        y = nose.y

        # Simple head direction
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

        cv2.putText(
            frame,
            direction,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Show camera
    cv2.imshow("Online Exam - Head Pose", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()