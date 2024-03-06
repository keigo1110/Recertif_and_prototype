'''
retinafaceで顔認識をするだけ．（リアルタイム）
'''

import cv2
from retinaface import RetinaFace
from PIL import Image
import numpy as np

def detect_faces(img):
    # PIL Image を NumPy 配列に変換
    img_np = np.array(img)

    # NumPy 配列を使用して顔を検出
    result = RetinaFace.detect_faces(img_path=img_np)

    # result が None または空の辞書でないことを確認
    if result is None or not result:
        return None

    # result が辞書型の場合のみ処理を進める
    if isinstance(result, dict):
        boxes = []
        for key, face in result.items():
            facial_area = face['facial_area']
            boxes.append(facial_area)
        return boxes
    else:
        return None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    boxes = detect_faces(frame_pil)

    if boxes is not None:
        for box in boxes:
            box = np.array(box).astype(int)
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 20)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
