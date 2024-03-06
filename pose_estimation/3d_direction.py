import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# MediaPipeとMatplotlibの初期化
mp_pose = mp.solutions.pose

# 一人のみを認識するための設定
pose = mp_pose.Pose()
# 複数人を認識するための設定(関数も変更する必要がある)
# pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 視点の初期値
elev = 270
azim = -270

def calculate_orientation_vector(landmarks):
    '''ランドマーク取得'''
    #左目外側の位置
    left_eye_inner = np.array([landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].y,
                               landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].z])
    #左目内側の位置
    left_eye_outer = np.array([landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].y,
                               landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].z])
    #右目外側の位置
    right_eye_inner = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y,
                                landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].z])
    #右目内側の位置
    right_eye_outer = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].y,
                                landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].z])
    # 口の左端の位置
    mouth_left = np.array([landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x,
                           landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y,
                           landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].z])
    # 口の右端の位置
    mouth_right = np.array([landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x,
                            landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y,
                            landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].z])

    #左肩の位置
    left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z])
    #右肩の位置
    right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z])
    #左腰の位置
    left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z])
    #右腰の位置
    right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z])
    #右膝の位置
    right_knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                           landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z])
    #左膝の位置
    left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z])
    #右足首の位置
    right_ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z])
    #左足首の位置
    left_ankle = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z])

    '''ベクトル計算'''
    # 胴体平面
    torso1 = np.cross(right_hip - right_shoulder, left_shoulder - right_shoulder)
    torso2 = np.cross(left_shoulder - left_hip, right_hip - left_hip)
    #胴体方向ベクトル
    troso = torso1 + torso2

    # 目線平面
    eyesight1 = np.cross(mouth_right - right_eye_outer, right_eye_inner - right_eye_outer)
    eyesight2 = np.cross(left_eye_inner - left_eye_outer, mouth_left - left_eye_outer)
    #目線方向ベクトル
    eyesight = eyesight1 + eyesight2

    #移動方向ベクトル
    move_vector = troso + eyesight

    # 正規化
    move_vector /= np.linalg.norm(move_vector)

    '''姿勢状態ベクトルを出す前に正規化'''
    right_composite_1 = right_shoulder - right_hip
    right_composite_1 /= np.linalg.norm(right_shoulder - right_hip)
    right_composite_2 = right_knee - right_hip
    right_composite_2 /= np.linalg.norm(right_knee - right_hip)
    right_composite = right_composite_1 + right_composite_2

    left_composite_1 = left_shoulder - left_hip
    left_composite_1 /= np.linalg.norm(left_shoulder - left_hip)
    left_composite_2 = left_knee - left_hip
    left_composite_2 /= np.linalg.norm(left_knee - left_hip)
    left_composite = left_composite_1 + left_composite_2

    # 姿勢状態ベクトル（腰を始点にした肩と膝の合成ベクトル）
    move_vector_length = np.linalg.norm(right_composite) + np.linalg.norm(left_composite)
    move_vector_length = 1 / move_vector_length

    '''
    #移動可能性ベクトルの長さ（肩と膝の距離から計算）
    move_vector_length = np.linalg.norm(right_shoulder - right_knee) + np.linalg.norm(left_shoulder - left_knee)+3*np.linalg.norm(right_ankle - left_ankle)
    '''

    return move_vector, move_vector_length


def plot_landmarks_with_orientation(landmarks, ax):
    global elev, azim

    if landmarks is not None:
        x = [lmk.x for lmk in landmarks]
        y = [-lmk.y for lmk in landmarks]
        z = [lmk.z for lmk in landmarks]

        min_y = min(y)
        y = [y_i - min_y for y_i in y]

        ax.clear()
        ax.scatter(x, y, z)

        for connection in mp.solutions.pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            ax.plot([x[start_idx], x[end_idx]], [y[start_idx], y[end_idx]], [z[start_idx], z[end_idx]], 'ro-')

        move_vector, move_vector_length = calculate_orientation_vector(landmarks)

        # 正規化された左右のヒップ座標
        left_hip_y = -landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y - min_y
        right_hip_y = -landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y - min_y

        hip_center = (np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, left_hip_y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]) +
                      np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, right_hip_y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z])) / 2.0

        arrow_length = move_vector_length
        #ax.quiver(*hip_center, *move_vector, length=arrow_length, color='green')
        ax.quiver(hip_center[0], hip_center[1], -hip_center[2], *move_vector, length=arrow_length, color='green')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 1.5)
        ax.set_zlim(0, 1.5)
        # 視点を変数で設定
        ax.view_init(elev=elev, azim=azim)

# カメラの初期化
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow("Pose Estimation", frame)

    if results.pose_landmarks:
        # 現在の視点を取得
        elev, azim = ax.elev, ax.azim
        plot_landmarks_with_orientation(results.pose_landmarks.landmark, ax)
        plt.pause(0.01)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.close()


















