'''
retinafaceで顔認識をするだけ．（画像）
'''
import cv2
from retinaface import RetinaFace
from PIL import Image
import numpy as np

def detect_faces_and_save_image(input_image_path, output_image_path):
    img = Image.open(input_image_path)
    img_np = np.array(img)
    result = RetinaFace.detect_faces(img_path=img_np)

    if result and isinstance(result, dict):
        for key, face in result.items():
            facial_area = face['facial_area']
            x1, y1, x2, y2 = np.array(facial_area).astype(int)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(output_image_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

# 使用例
detect_faces_and_save_image('/Users/keigo/Downloads/Screenshot 2023-12-27 at 10.20.42.jpg', '/Users/keigo/Downloads/output_image3.jpg')
