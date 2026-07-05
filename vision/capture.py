import cv2
import os

os.makedirs("calibration", exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        cv2.imwrite(f"calibration/img_{count:02d}.jpg", frame)
        print(f"Saved {count}")
        count += 1

    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()