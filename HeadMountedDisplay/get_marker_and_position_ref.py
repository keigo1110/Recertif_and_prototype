import cv2
from cv2 import aruco
import numpy as np
import yaml
from picamera2 import Picamera2
from libcamera import controls

class MarkSearch:
    def __init__(self, aruco_dict, parameters, calibration_data):
        self.aruco_dict = aruco_dict
        self.parameters = parameters
        self.camera_matrix = np.array(calibration_data["camera_matrix"])
        self.distortion_coefficients = np.array(calibration_data["distortion_coefficients"])

    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        return corners, ids

    def draw_markers(self, frame, corners, ids):
        if len(corners) > 0:
            frame = aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.035, self.camera_matrix, self.distortion_coefficients)

            for i in range(len(ids)):
                aruco.drawAxis(frame, self.camera_matrix, self.distortion_coefficients, rvecs[i], tvecs[i], 0.1)
                marker_position = tvecs[i].ravel()
                print(f"Marker ID {ids[i][0]} 3D position: x={marker_position[0]}, y={marker_position[1]}, z={marker_position[2]}")

    def get_mark_center(self, num_id, corners, ids):
        if num_id in np.ravel(ids):
            index = np.where(ids == num_id)[0][0]
            cornerUL, cornerBR = corners[index][0][0], corners[index][0][2]
            center = [(cornerUL[0] + cornerBR[0]) / 2, (cornerUL[1] + cornerBR[1]) / 2]
            print(f'Center of marker ID {num_id}: {center}')
            return center


# Load calibration data
with open("calibration_data.yaml", "r") as file:
    calibration_data = yaml.safe_load(file)

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

# Initialize marker detection
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
mark_search = MarkSearch(aruco_dict, parameters, calibration_data)

while True:
    frame = picam2.capture_array()
    frame = cv2.flip(frame, -1)

    if frame is None or frame.size == 0:
        print("Empty frame")
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    corners, ids = mark_search.detect_markers(frame)
    mark_search.draw_markers(frame, corners, ids)
    
    markID = 0
    mark_search.get_mark_center(markID, corners, ids)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
