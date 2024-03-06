'''
retinafaceで顔認識をするだけ．（動画）
'''
import cv2
from retinaface import RetinaFace
from PIL import Image
import numpy as np

def detect_faces_in_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        img_np = np.array(frame_pil)
        result = RetinaFace.detect_faces(img_path=img_np)

        if result and isinstance(result, dict):
            for key, face in result.items():
                facial_area = face['facial_area']
                x1, y1, x2, y2 = np.array(facial_area).astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

# 使用例
detect_faces_in_video('input_video.mp4', 'output_video.mp4')
