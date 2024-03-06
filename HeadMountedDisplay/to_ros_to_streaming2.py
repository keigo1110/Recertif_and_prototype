import cv2
from cv2 import aruco
import numpy as np
import yaml
import json
import websocket
from picamera2 import Picamera2
from libcamera import controls
from bottle import route, run, response
import os

# キャリブレーションデータのロード
with open("calibration_data.yaml", "r") as file:
    calibration_data = yaml.safe_load(file)

camera_matrix = np.array(calibration_data['camera_matrix'])
distortion_coefficients = np.array(calibration_data['distortion_coefficients'])

# マーカー検出の初期化
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

# カメラの初期化
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)}))
picam2.start()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

# WebSocketクライアントの初期化と接続の永続化
ws = None

# すでに描画されたマーカーIDを保存するリスト
drawn_ids = []

def establish_connection():
    global ws
    try:
        ws = websocket.WebSocket()
        ws.connect("ws://10.42.0.1:9090")
    except Exception as e:
        print("WebSocket接続エラー: ", e)
        ws = None

def publish_to_ros(topic, message):
    global ws
    if ws is None:
        establish_connection()
        if ws is None:
            return  # 接続できない場合は処理をスキップ

    ros_message = json.dumps({
        "op": "publish",
        "topic": topic,
        "msg": message
    })
    try:
        ws.send(ros_message)
    except Exception as e:
        print("WebSocket送信エラー: ", e)
        establish_connection()

# カメラとWebSocketの初期化
establish_connection()

def get_frame():
    frame = picam2.capture_array() # 画像を取得
    frame = cv2.flip(frame, -1) # 画像を上下反転
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) # 画像をBGRに変換

    # マーカーを検出して描画 (この部分も必要に応じて調整)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    # Define the offset variables
    width_offset = 150  # normal200
    height_offset = 150  # normal200

    # Process each marker if ids are not None
    if ids is not None:
        for i in range(len(ids)):
            # このIDがすでに描画されていたらスキップ
            if ids[i] in drawn_ids:
                continue
            # このIDを描画済みリストに追加
            drawn_ids.append(ids[i][0])

            # Estimate the marker's pose in camera coordinates
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.034, camera_matrix, distortion_coefficients)

            # Draw the marker's pose on the image
            aruco.drawAxis(frame, camera_matrix, distortion_coefficients, rvecs[i], tvecs[i], 0.1)

            # Draw bounding box around the marker
            corner = corners[i][0]
            top_left = tuple((corner[0] + [width_offset, height_offset]).astype(int))
            bottom_right = tuple((corner[2] - [width_offset, height_offset]).astype(int))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            # Draw the marker ID above the bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            font_color = (0, 0, 255)
            font_thickness = 2
            #text_position = (top_left[0], top_left[1] - 10)
            text_position = (bottom_right[0], bottom_right[1])

            # Draw the marker ID above the bounding box
            name = None
            if ids[i][0] == 0:  # 修正: idsの要素は配列になっているため
                name = "_Minami_robot"
            elif ids[i][0] == 1:
                name = "_Higashi_robot"

            if name:
                cv2.putText(frame, str(ids[i][0]) + name, text_position, font, font_scale, font_color, font_thickness)

    # マーカーが検出された場合
    if len(corners) > 0:
        # マーカーの位置をカメラ座標で算出
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.034, camera_matrix, distortion_coefficients)

        # 各マーカーを処理
        for i in range(len(ids)):
            # マーカーの位置を表示
            print(f"マーカーID: {ids[i]}")
            print(f"位置ベクトル: \n{tvecs[i]}")
            print(f"回転ベクトル: \n{rvecs[i]}\n")

            if ids[i] == 1:
                print("1")
                tvecs[i][0][0] += 10000
                print(tvecs[i][0][0])

            if ids[i] == 2:
                print("2")
                tvecs[i][0][0] += 20000
                print(tvecs[i][0][0])

            # マーカーの位置をROSに公開
            position_message = {

                "x": float(tvecs[i][0][0]),
                "y": float(tvecs[i][0][1]),
                "z": float(tvecs[i][0][2])
            }
            publish_to_ros("/marker_position", position_message)

            # 画像にマーカーの位置を描画
            aruco.drawAxis(frame, camera_matrix, distortion_coefficients, rvecs[i], tvecs[i], 0.1)
    else:
        # マーカーの位置をROSに公開
        position_message = {

            "x": float(0.0),
            "y": float(0.0),
            "z": float(0.0)
        }
        publish_to_ros("/marker_position", position_message)

    # このフレームでの描画が終わったらリストをクリア
    drawn_ids.clear()
    return frame

@route('/', method="GET")
def top():
    s = str(1920) + ' x ' + str(1080)
    return '<font size="8">{}</font><br><img src="/streaming"/>'.format(s)

@route('/streaming')
def streaming():
    response.set_header('Content-type', 'multipart/x-mixed-replace;boundary=--frame')

    while True:
        frame = get_frame()
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        if not ret:
            continue
        yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


if __name__ == '__main__':
    try:
        run(
            host='10.42.0.26',
            port=8080,
            reloader=False, # set to False to avoid second init of camera
            debug=True)
    finally:
        if ws is not None:
            ws.close()
        picam2.stop()
        print("exit")

