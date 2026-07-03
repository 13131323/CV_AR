import cv2
import numpy as np
import glob
from pathlib import Path

CHECKERBOARD = (9, 6)

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
# 기존 코드: objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
square_size = 19.0  # mm 단위
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

objpoints = []
imgpoints = []

BASE_DIR = Path(__file__).resolve().parent
CALIB_DIR = BASE_DIR.parent / "calibration"

images = [str(p) for p in CALIB_DIR.glob("*.jpg")]

print(CALIB_DIR)
print(f"Found {len(images)} images")

image_size = None  # 이미지 크기를 저장할 변수 초기화
success = 0

for fname in images:
    img = cv2.imread(fname)
    if img is None:  # 이미지가 제대로 로드되지 않으면 건너뜀
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 첫 번째 올바른 이미지에서 해상도(가로, 세로)를 딱 한 번만 저장
    if image_size is None:
        image_size = gray.shape[::-1]

    ret, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        success += 1

        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11,11),
            (-1,-1),
            criteria,
        )

        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500) # 0.5초 동안 확인

    else:
        print(f"Chessboard not found: {fname}")

cv2.destroyAllWindows()

print(f"Detected {success}/{len(images)} images")

# 에러 방지: 유효한 이미지가 없거나 크기를 지정하지 못했을 경우 예외 처리
if image_size is None or len(objpoints) == 0:
    print("Error: 이미지를 찾을 수 없거나 정상적인 이미지 파일이 아닙니다.")
else:
    # gray.shape[::-1] 대신 안전하게 저장된 image_size를 사용
    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )
    print(f"Calibration RMS Error: {ret:.4f}")

    print("--- Camera Matrix ---")
    print(cameraMatrix)
    print("\n--- Distortion Coefficients ---")
    print(dist)
    print("--------------------------------")

    np.savez(
        "camera_calibration.npz",
        cameraMatrix=cameraMatrix,
        distCoeffs=dist
    )

    print("Calibration saved to camera_calibration.npz")

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print(f"Total error: {mean_error / len(objpoints)}")