import open3d as o3d
import json
import json5
import numpy as np
import base64 # バイナリデータをテキスト形式に変換するライブラリ
from websocket import WebSocketApp
import matplotlib.pyplot as plt
import time

from sklearn.neighbors import NearestNeighbors

received_message = None

instances = {}  # マーカーIDをキーとしてインスタンスを保持する辞書

class Recertif:
    def __new__(cls, box_id, corner_points, marker_id):
        if marker_id in instances:
            instance = instances[marker_id]
            instance.update(box_id, corner_points)
            return instance

        return super().__new__(cls)

    def __init__(self, box_id, corner_points, marker_id):
        if marker_id in instances:
            return

        self.box_id = box_id
        self.corner_points = np.array(corner_points)
        self.marker_id = marker_id

        instances[marker_id] = self

    def __str__(self):
        return (f"box_id: {self.box_id}, "
                f"corner_points: {self.corner_points.tolist()}, "
                f"marker_id: {self.marker_id}")

    @classmethod
    def update_by_box_id(cls, box_id, corner_points):
        # box_idを基にインスタンスを検索してアップデートするクラスメソッド
        instance = next((i for i in instances.values() if i.box_id == box_id), None) # box_idが一致するインスタンスを取得
        if instance:
            instance.update(box_id, corner_points) # インスタンスをアップデート
        else:
            print(f"No instance found for box_id {box_id}")

    def update(self, box_id, corner_points):
        #print(str(instances[self.marker_id]) + "をアップデートします")
        if self.box_id == box_id:
            self.corner_points = np.array(corner_points)
            center = np.mean(self.corner_points, axis=0)  # corner_pointsの中心点を計算
            if self.marker_id == 0 or self.marker_id == 1 or self.marker_id == 2:
                processor.ax.text(center[0], center[1], center[2] + 1, "robot_" + str(self.marker_id), color='red', weight="bold", fontsize=36)  # 中心点にIDを描画、フォントサイズ指定できたっけ？
            else:
                processor.ax.text(center[0], center[1], center[2] + 1, "human_" + str(self.marker_id), color='blue', weight="bold", fontsize=36)
            #plt.pause(0.01)  # プロットを表示
            processor.fig.canvas.draw_idle()

            #print(str(instances[self.marker_id]) + "をアップデートしました")
        else:
            del instances[self.marker_id]
            #print("インスタンスを削除しました")

class KalmanFilter:
    def __init__(self, dim_x, dim_z, dim_u):
        # 状態変数の次元
        self.dim_x = dim_x
        # 観測変数の次元
        self.dim_z = dim_z
        # 制御変数の次元
        self.dim_u = dim_u

        # 状態変数
        self.x = np.zeros((dim_x, 1))
        # 状態の不確実性を表す共分散行列
        self.P = np.eye(dim_x) * 1000

        # 状態遷移行列
        self.F = np.eye(dim_x)
        # 制御入力行列
        self.B = np.zeros((dim_x, dim_u))
        # 観測行列
        self.H = np.eye(dim_z, dim_x)

        # システムノイズ、観測ノイズの共分散行列
        self.Q = np.eye(dim_x) * 0.1
        self.R = np.eye(dim_z) * 0.1

    def predict(self, u=0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.dim_x)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

class PointCloudProcessor:
    def __init__(self, config_file):
        self.bounding_boxes = {}  # corner_pointsをbox_idにマッピングする辞書
        self.run_ar_triggered = False  # run_arからの取得があったかどうかのフラグ

        with open(config_file) as f:
            config = json5.load(f) # ファイルの内容をJSONとして読み込み、Pythonの辞書に変換

        # config辞書から各設定値を読み込み、インスタンス変数に保存します。
        self.filter_threshold = config['filter_threshold']
        self.current_elev = config['elev']
        self.current_azim = config['azim']
        self.ws_url = config['ws'] #デスクは230,ノートは136
        self.topic = config['topic']
        self.plot_settings = config['plot_settings']
        self.nb_neighbors = config['nb_neighbors']
        self.std_ratio = config['std_ratio']
        self.nb_points = config['nb_points']
        self.radius = config['radius']
        self.distance_threshold = config['distance_threshold']
        self.ransac_n = config['ransac_n']
        self.num_iterations = config['num_iterations']
        self.eps = config['eps']
        self.min_points = config['min_points']
        self.max_points = config['max_points']

        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection = '3d')

        # 前フレームのバウンディングボックスを保存する辞書
        self.prev_bounding_boxes = {}
        self.current_id = 0  # バウンディングボックスに一意のIDを割り当てるための変数

        # バウンディングボックスのIDとカルマンフィルタのマッピング
        self.kalman_filters = {}

    def assign_id_to_new_box(self, min_points, max_points):
        min_distance = float("inf")
        best_match_id = None

        for box_id, (prev_min, prev_max) in self.prev_bounding_boxes.items():
            distance = np.linalg.norm((min_points + max_points) - (prev_min + prev_max))

            if distance < self.filter_threshold:  # 一定範囲内であれば同一物体と判断（この値は調整が必要）
                if distance < min_distance:
                    min_distance = distance
                    best_match_id = box_id

        if best_match_id is None:
            new_id = self.current_id
            self.current_id += 1
            # 新しいIDでカルマンフィルタを初期化
            self.kalman_filters[new_id] = KalmanFilter(dim_x=6, dim_z=3, dim_u=0)
            return new_id
        else:
            return best_match_id

    def remove_detected_planes(self, pcd):
        """使用したdetect_planar_patchesメソッドで平面を検出し、除去します。"""
        # 平面パッチの検出
        oboxes = pcd.detect_planar_patches(
            normal_variance_threshold_deg=70, #default = 60
            coplanarity_deg=80, # default = 75
            outlier_ratio=0.80, # default = 0.75
            min_plane_edge_length=0, # default = 0
            min_num_points=0, # default = 0
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))

        # 検出された平面パッチに関連する点を除去
        all_indices_to_remove = []
        for obox in oboxes:
            inside_box = obox.get_point_indices_within_bounding_box(pcd.points)
            all_indices_to_remove.extend(inside_box)

        return pcd.select_by_index(all_indices_to_remove, invert=True)

    #x_threshold=(-2.2, 2.0), y_threshold=(-3.6, 3.9), z_threshold=2.0
    # 区切る
    def filter_points_based_on_xyz(self, pcd, x_threshold=(-0.5, 0.5), y_threshold=(0.0, 3.0), z_threshold=2.0):
        """x, y, z 座標に基づいて点群をフィルタリングする"""
        points = np.asarray(pcd.points)
        valid_indices = np.where(
            (points[:, 0] > x_threshold[0]) & (points[:, 0] < x_threshold[1]) &
            (points[:, 1] > y_threshold[0]) & (points[:, 1] < y_threshold[1]) &
            (points[:, 2] < z_threshold)
        )[0]
        return pcd.select_by_index(valid_indices)

    def augment_point_cloud(self, pcd, k=5, new_points_per_original_point=5):
        points = np.asarray(pcd.points)
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points)
        distances, indices = nbrs.kneighbors(points)

        new_points = []
        for i in range(len(points)):
            neighbors = indices[i][1:]

            for _ in range(new_points_per_original_point):
                weights = np.random.rand(k)
                weights /= weights.sum()  # Normalize weights to make their sum 1

                weighted_neighbors = weights.reshape(-1, 1) * points[neighbors]
                new_point = weighted_neighbors.sum(axis=0)
                new_points.append(new_point)

        augmented_points = np.vstack((points, new_points))
        augmented_pcd = o3d.geometry.PointCloud()
        augmented_pcd.points = o3d.utility.Vector3dVector(augmented_points)

        return augmented_pcd


    def process_point_cloud(self, points):
        # 空のPointCloudオブジェクトを生成
        pcd = o3d.geometry.PointCloud()

        # 引数で受け取った点群データをPointCloudオブジェクトにセット
        pcd.points = o3d.utility.Vector3dVector(points)

        # 他の前処理の後でz座標に基づいて点群をフィルタリング
        pcd = self.filter_points_based_on_xyz(pcd)

        # 点群の法線を計算
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # 統計的外れ値除去
        pcd, _ = pcd.remove_statistical_outlier(self.nb_neighbors, self.std_ratio)

        # 放射状外れ値除去
        #pcd, _ = pcd.remove_radius_outlier(self.nb_points, self.radius)

        # 新しい平面除去のメソッドを使用
        pcd = self.remove_detected_planes(pcd)

        # 点群をアップサンプリングする
        pcd = self.augment_point_cloud(pcd)

        # 平面除去
        _, inliers = pcd.segment_plane(self.distance_threshold, self.ransac_n, self.num_iterations)
        #print("inliers:")
        #print(inliers)

        # 処理した点群を返す
        return pcd.select_by_index(inliers, invert=True)


    def plot_point_cloud(self, pcd, labels):
        # 現在のプロットをクリア
        self.ax.cla()

        # 点群データをX, Y, Z座標に分解
        x, y, z = np.array(pcd.points).T

        # PointCloudに色情報がある場合はそれを使用。ない場合は青色を使用
        colors = np.asarray(pcd.colors) if pcd.has_colors() else 'b'

        # カメラの視点（仰角と方位角）を設定
        self.ax.view_init(elev=self.current_elev, azim=self.current_azim)

        # 点群をプロット
        self.ax.scatter(x, y, z, c=colors, marker='o', s=self.plot_settings['marker_size'])

        # 軸ラベルの設定
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # 軸の範囲を設定
        self.ax.set_xlim(self.plot_settings['xlim'])
        self.ax.set_ylim(self.plot_settings['ylim'])
        self.ax.set_zlim(self.plot_settings['zlim'])

        self.draw_bounding_box_for_segments(pcd, labels)

        #プロットを表示
        #plt.pause(0.01)
        self.fig.canvas.draw_idle()

    def plot_point_cloud_ar(self, x, y, z, color='r'):
        if x != 0 and y != 0 and z != 0:
            # カメラの視点（仰角と方位角）を設定
            self.ax.view_init(elev=self.current_elev, azim=self.current_azim)
            # self.ax.scatter(x, y, z, c=color, marker='o', s=self.plot_settings['marker_size'])
            self.ax.scatter(x, y, z, c=color, marker='o', s=30) # マーカーのサイズを変更（元は20）
            self.ax.set_xlabel('X') # カメラの左右
            self.ax.set_ylabel('Y') # カメラの上下
            self.ax.set_zlabel('Z') # カメラからの奥行
            self.ax.set_xlim(self.plot_settings['xlim'])
            self.ax.set_ylim(self.plot_settings['ylim'])
            self.ax.set_zlim(self.plot_settings['zlim'])
            #plt.pause(0.01)
            self.fig.canvas.draw_idle()

    def segment_using_dbscan(self, pcd):
        eps = self.eps
        min_points = self.min_points
        max_points = self.max_points

        # DBSCANでクラスタリング
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

        # 最大のラベル（クラスタID）を取得
        max_label = labels.max()

        # 各セグメントのラベルと点の数を表示
        unique_labels, counts = np.unique(labels, return_counts=True)

        # セグメントごとの点を保存する辞書を作成
        segmented_points = {}
        for label, count in zip(unique_labels, counts):
            if label == -1:  # ノイズは無視
                continue
            if count > max_points:  # 閾値を超える場合も無視
                print(f"Ignoring label {label} with {count} points")
                continue
            #print(f"Label: {label}, Number of points: {count}")
            segmented_points[label] = np.asarray(pcd.points)[labels == label]

        # ノイズ（ラベル-1）を除外し、それ以外の点を抽出
        non_noise_labels = labels[labels != -1]
        non_noise_points = np.asarray(pcd.points)[labels != -1]

        # 各クラスタに色を割り当てるためのカラーマップを生成
        colors = plt.cm.jet(np.linspace(0, 1, max_label + 1))

        # ポイントごとに色を割り当てる
        point_colors = colors[non_noise_labels]

        # ノイズを除去した点と色情報で新しいPointCloudを生成
        segmented_pcd = o3d.geometry.PointCloud()
        segmented_pcd.points = o3d.utility.Vector3dVector(non_noise_points)
        segmented_pcd.colors = o3d.utility.Vector3dVector(point_colors[:, :3]) # 最後の要素は透明度なので除外

        print("Unique labels:", np.unique(labels))
        print(segmented_pcd)

        return segmented_pcd, segmented_points

    def draw_bounding_box(self, pcd):
        points = np.asarray(pcd.points)
        min_points = np.min(points, axis=0)
        max_points = np.max(points, axis=0)

        # Bounding Boxの寸法
        dx, dy, dz = max_points - min_points

        # Bounding Boxの角の座標
        corner_points = [
            min_points,
            min_points + [dx, 0, 0],
            min_points + [dx, dy, 0],
            min_points + [0, dy, 0],
            min_points + [0, 0, dz],
            min_points + [dx, 0, dz],
            min_points + [dx, dy, dz],
            min_points + [0, dy, dz]
        ]

        # 角から線を描画
        for start, end in [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]:
            xs, ys, zs = zip(*[corner_points[start], corner_points[end]])
            self.ax.plot(xs, ys, zs, c='r')
        #print("Corner points:", corner_points)
        return corner_points


    def draw_bounding_box_for_segments(self, pcd, labels):
        unique_labels = np.unique(labels)
        new_bounding_boxes = {}

        for label in unique_labels:
            if label == -1:
                continue

            segment_points = np.asarray(pcd.points)[labels == label]
            min_points = np.min(segment_points, axis=0)
            max_points = np.max(segment_points, axis=0)

            box_id = self.assign_id_to_new_box(min_points, max_points)

            # バウンディングボックスの中心点を計算
            center = (min_points + max_points) / 2.0

            # 状態（位置）を更新
            self.kalman_filters[box_id].update(np.array([center]).T)
            # 予測ステップ
            self.kalman_filters[box_id].predict()

            # このセグメントに対するバウンディングボックスを描画
            segment_pcd = o3d.geometry.PointCloud()
            segment_pcd.points = o3d.utility.Vector3dVector(segment_points)
            corner_points = self.draw_bounding_box(segment_pcd) # corner_pointsを取得

            # corner_pointsをbox_idにマッピングして保存
            self.bounding_boxes[box_id] = corner_points

            # 中心点にIDを描画
            self.ax.text(center[0], center[1], center[2], str(box_id), color='lime', weight="bold")

            #print(f"Assigned ID {box_id} to bounding box")
            new_bounding_boxes[box_id] = (min_points, max_points)

        # 前のフレームに存在して現在のフレームには存在しないバウンディングボックスのIDを削除
        for box_id in self.prev_bounding_boxes.keys(): # 前のフレームのバウンディングボックスのIDを取得
            if box_id not in new_bounding_boxes: # 前のフレームのバウンディングボックスのIDが現在のフレームに存在しない場合
                del self.bounding_boxes[box_id] # bounding_boxesから削除
                del self.kalman_filters[box_id] # kalman_filtersから削除
                print(f"Removed ID {box_id}") # 削除したことを表示

        self.prev_bounding_boxes = new_bounding_boxes
        self.new_bounding_boxes = new_bounding_boxes

    def is_point_inside_bounding_box(self, point, corner_points):
        corner_points = np.array(corner_points) # リストをNumPy配列に変換
        min_points = np.min(corner_points, axis=0) # 各軸の最小値を計算
        max_points = np.max(corner_points, axis=0) # 各軸の最大値を計算

        x, y, z = point # pointの座標を取得
        x_min, y_min, z_min = min_points
        x_max, y_max, z_max = max_points

        return x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max

    # WebSocketからメッセージを受け取ったときの処理
    def on_message(self, ws, message):
        time1 = time.time()
        # 現在の視点（elevationとazimuth）を保存
        self.current_elev, self.current_azim = self.ax.elev, self.ax.azim

        # 受信したメッセージデータからデータを解析
        data = json.loads(message)
        byte_data = np.frombuffer(base64.b64decode(data['msg']['data']), dtype=np.uint8)
        points_np = byte_data.view(dtype=np.float32).reshape((-1, 8))
        points = points_np[:, :3]

        # 前処理
        processed_pcd = self.process_point_cloud(points)

        # DBSCANによるセグメンテーション
        segmented_pcd, segmented_points = self.segment_using_dbscan(processed_pcd)
        labels = np.array(segmented_pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points, print_progress=True))

        # 更新された点群データを描画
        self.plot_point_cloud(segmented_pcd, labels)

        # 更新されたarucoマーカの座標を描画
        self.run_ar_triggered = True  # run_arがデータを取得したことを示す
        # ARからのデータを取得したデータを描画
        if self.run_ar_triggered:  # run_arからの取得があった場合
            #print("取得したよ〜")
            self.run_ar()
            self.run_ar_triggered = False  # フラグをリセット

        for box_id, corner_points in self.bounding_boxes.items():
            # ここで全てのbox_idの中からインスタンス化されているものを取り出して，アップデートメソッドを呼び出す
            for instance in instances.values():
                # アップデートメソッドを呼び出す
                instance.update_by_box_id(box_id, corner_points)

        # 可視化する（ここが唯一）
        plt.pause(0.01)
        time2 = time.time()
        print(time2-time1)


    # この関数内で、arucoマーカの座標とバウンディングボックスの座標を使用して判定を行います。
    def on_message_ar(self, ws_ar, message_ar):
        global received_message

        received_message = json.loads(message_ar)
        # print("Received message:", received_message)
        if 'msg' in received_message:
            point = received_message['msg']
            x, y, z = point['x'], point['y'], point['z']

            if x != 0 and y != 0 and z != 0:
                # 以下はarucoマーカの座標変換のコード
                x_angle = np.radians(285) # もともとは270
                r_x = np.array([
                    [1, 0, 0],
                    [0, np.cos(x_angle), -np.sin(x_angle)],
                    [0, np.sin(x_angle), np.cos(x_angle)]
                ])

                transformed_point = np.dot(r_x, np.array([x, y, z]))
                x_new, y_new, z_new = transformed_point
                # マーカ位置
                x_new = x_new - 0.1
                y_new = y_new - 0.9
                z_new = z_new + 0.4 # もともとは1.56

                if 9900 < x_new and x_new < 19000:
                    marker_id = 1
                    x_new = x_new - 10000
                elif 19000 < x_new and x_new < 29000:
                    marker_id = 2
                    x_new = x_new - 20000
                elif 29000 < x_new and x_new < 39000:
                    marker_id = 3
                    x_new = x_new - 30000
                elif x_new < 9900:
                    marker_id = 0

                print(f"marker_idはこれです！: {marker_id}")

                # 可視化関数
                if marker_id == 0 or marker_id == 1 or marker_id == 2:
                    self.plot_point_cloud_ar(x_new, y_new, z_new, color='k') # もとはr
                elif marker_id == 3:
                    self.plot_point_cloud_ar(x_new, y_new, z_new, color='b')

                # ここでバウンディングボックスのidと座標を取得
                for box_id, corner_points in self.bounding_boxes.items():
                    # print(f"Checking bounding box ID {box_id} with corner points {corner_points}")

                    # arucoマーカの座標がバウンディングボックス内にあるかどうかを判定
                    if self.is_point_inside_bounding_box((x_new, y_new, z_new), corner_points):
                        # もしマーカがバウンディングボックスの中にあったら
                        #print(f"The aruco marker is inside the bounding box ID {box_id}.")
                        #print(f"{box_id}検知したよ〜")
                        #マーカのIDの検知とバウンディングボックスIDの統合を行う
                        _ = self.bb_marker_integration(box_id, corner_points, marker_id)
                        break
                    else:
                        print(f"The aruco marker is outside the bounding box ID {box_id}.")

            ws_ar.close()

    # マーカのIDの検知とバウンディングボックスIDの統合を行う関数
    def bb_marker_integration(self, box_id, corner_points, marker_id):
        # 識別子保持
        identify_info = Recertif(box_id, corner_points, marker_id)
        return identify_info

    # WebSocketが開いたときの処理
    def on_open(self, ws):
        msg = {
            "op": "subscribe",
            "topic": self.topic,
            "type": "sensor_msgs/PointCloud2"
        }
        ws.send(json.dumps(msg))

    # WebSocketが開いたときの処理
    def on_open_ar(self, ws_ar):
        msg_ar = {
            "op": "subscribe",
            "topic": "/marker_position",
            "type": "geometry_msgs/Point"
        }
        ws_ar.send(json.dumps(msg_ar))

    # WebSocketでエラーが発生したときの処理
    def on_error(self, ws, error):
        print(f"An error occurred: {error}")

    def run(self): # WebSocketを開始
        ws = WebSocketApp(self.ws_url, # WebSocketサーバのURL
                          on_open=self.on_open, # WebSocketが開いたときの処理
                          on_message=self.on_message, # メッセージを受け取ったときの処理
                          on_error=self.on_error) # エラーが発生したときの処理
        ws.run_forever()

    def run_ar(self):
        ws_ar = WebSocketApp(self.ws_url,
                            on_open=self.on_open_ar,
                            on_message=self.on_message_ar,
                            on_error=self.on_error)
        plt.show(block=False)
        ws_ar.run_forever()


if __name__ == "__main__":
    processor = PointCloudProcessor(config_file='config.json5')
    processor.run()





