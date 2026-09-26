import cv2

# 1. Open the webcam
cap = cv2.VideoCapture(0)

while True:

    # 2. Continuously read frames
    success, frame = cap.read()

    if not success:
        print("Failed to read webcam")
        break

    # 3. Display the webcam
    cv2.imshow("Webcam", frame)

    # 4. Stop when you press q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 5. Release the camera properly
cap.release()
cv2.destroyAllWindows()
