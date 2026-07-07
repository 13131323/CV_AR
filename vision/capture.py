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
    
    display = cv2.flip(frame, 1)  # 화면만 좌우 반전

    cv2.imshow("Camera", display)

    key = cv2.waitKey(1)

    if key == ord('s'):
        cv2.imwrite(f"calibration/img_{count:02d}.jpg", frame) # 원본 저장
        print(f"Saved {count}")
        count += 1

    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()