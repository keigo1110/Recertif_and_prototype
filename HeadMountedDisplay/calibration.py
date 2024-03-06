import numpy as np
import cv2
import glob

# 格子の交点数
CHECKERBOARD = (6,9)

# 格子の交点の3D座標を保存する配列
objpoints = []
# 格子の交点の2Dイメージ座標を保存する配列
imgpoints = [] 

# 3Dの参照ポイントを生成（z = 0）
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# キャリブレーション画像を読み込む
images = glob.glob('images/*.jpg')  # 実際のパスに変更してください

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # チェスボードのコーナーを検出
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    # コーナーが見つかった場合、objpoints, imgpointsに追加
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

# 画像が一枚も検出されなかった場合のエラーハンドリング
if len(imgpoints) == 0:
    print("No corners detected. Please check the images or the checkerboard pattern.")
else:
    # カメラキャリブレーションを実行
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # カメラマトリックス、歪み係数、回転ベクトル、平行移動ベクトルを表示
    print("Camera Matrix: ")
    print(mtx)
    print("\nDistortion Coefficients: ")
    print(dist)
    print("\nRotation Vectors: ")
    print(rvecs)
    print("\nTranslation Vectors: ")
    print(tvecs)
