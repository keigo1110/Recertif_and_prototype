# Recertif

このPythonスクリプトは、提案手法Recertifに関するプログラムがあります。

## 必要条件

実行する前に、必要なライブラリがインストールされていることを確認してください。
以下のコマンドを使用してインストールできます。

```bash
pip install -r requirements.txt
```

## 使用方法
### PC1
スクリプトを実行するには、まずROSを立ち上げます
```bash
roscore
```
そして、[LiDAR](https://github.com/keigo1110/unilidar_sdk)を起動します。
```bash
/dev/ttyUSB0
```
Build
```bash
cd unilidar_sdk/unitree_lidar_ros
catkin_make
```
RUN
```bash
source devel/setup.bash
roslaunch unitree_lidar_ros run.launch
```
可視化の必要がある場合はRvizを立ち上げます。
```bash
rviz
```
次に、rosbridgeを起動します。
```bash
roslaunch rosbridge_server rosbridge_websocket.launch
```
### PC2
そして、ヘッドマウントディスプレイ用のraspberry piで、
```bash
python script_name.py
```
用意したスクリプトを実行します。
`script_name.py`を実際のスクリプトファイル名に置き換えてください。

### PC3
これらを起動した状態で、Recertifのプログラムを実行します。
```bash
python script_name.py
```
`script_name.py`を実際のスクリプトファイル名に置き換えてください。

## 注意事項
ネットワークのアドレスには気をつけてください。

### ROSのwifi設定
```bash
nano setup.bash
```
このファイルの最後の2行を自身のIPアドレスにしてください。
```bash
source .bashrc
```
これで設定完了です。

### rosbagの再生
```bash
rosbag play bagファイル名.bag -l
```
### rvizの設定ファイルで実行
```bash
rviz -d 設定ファイル名.rviz
```
