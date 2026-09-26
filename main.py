import cv2
import time

from head_pose import detect_head_pose


# ==========================================
# SETTINGS
# ==========================================

CAMERA_INDEX = 0

WARNING_TIME = 2        # seconds
MAX_WARNINGS = 3


# ==========================================
# START CAMERA
# ==========================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


# ==========================================
# VARIABLES
# ==========================================

warning_count = 0

looking_away = False
away_start_time = None

last_direction = "CENTER"


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        print("Error: Could not read frame.")
        break


    # ======================================
    # HEAD POSE
    # ======================================

    direction = detect_head_pose(frame)

    if direction is None:
        direction = "NO FACE"


    # ======================================
    # CHECK HEAD DIRECTION
    # ======================================

    if direction in ["LOOKING LEFT", "LOOKING RIGHT",
                     "LOOKING UP", "LOOKING DOWN"]:

        if not looking_away:

            looking_away = True
            away_start_time = time.time()

        else:

            elapsed_time = time.time() - away_start_time

            # ------------------------------
            # Student looking away
            # ------------------------------

            if elapsed_time >= WARNING_TIME:

                warning_count += 1

                print(
                    f"WARNING {warning_count}: "
                    f"Student is {direction}"
                )

                # Reset timer
                looking_away = False
                away_start_time = None

                # --------------------------
                # Maximum warnings
                # --------------------------

                if warning_count >= MAX_WARNINGS:

                    print("Maximum warnings reached!")

    else:

        looking_away = False
        away_start_time = None


    # ======================================
    # DISPLAY INFORMATION
    # ======================================

    cv2.putText(
        frame,
        f"Direction: {direction}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Warnings: {warning_count}/{MAX_WARNINGS}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )


    # ======================================
    # WARNING ON SCREEN
    # ======================================

    if looking_away:

        elapsed_time = time.time() - away_start_time

        cv2.putText(
            frame,
            f"WARNING: LOOKING AWAY {elapsed_time:.1f}s",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ======================================
    # SHOW CAMERA
    # ======================================

    cv2.imshow(
        "Online Exam Proctoring System",
        frame
    )


    # ======================================
    # PRESS Q TO EXIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# RELEASE CAMERA
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Exam monitoring stopped.")
print(f"Total warnings: {warning_count}")