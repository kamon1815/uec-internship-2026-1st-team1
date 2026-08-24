#MiDaSを用いた深度推定のサンプルプログラム

import cv2
import torch
import numpy as np
from ultralytics import YOLO

# 画像のパスを指定
image_path = cv2.VideoCapture(0)

# 画像読み込み


# MiDaSで深度推定モデル読み込み
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to("cpu").eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# YOLOv8モデル読み込み（人物検出）
model = YOLO("yolov8n.pt")

# --- キャリブレーション仮定値 ---
scale_factor = 3.0  # 相対深度をメートル換算


while image_path.isOpened():
    success,frame = image_path.read()
    if success:
        # 1. 深度マップ推定
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = midas_transforms(img_rgb)
        with torch.no_grad():
            prediction = midas(input_tensor)
            depth_map = prediction.squeeze().cpu().numpy()
            depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

        # 深度マップの正規化
        depth_min, depth_max = depth_map.min(), depth_map.max()
        depth_map_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)

        # 仮のカメラ内部パラメータ
        fx = 500.0
        fy = 500.0
        cx = frame.shape[1] / 2
        cy = frame.shape[0] / 2

        # 2. 人物検出
        results = model(frame)[0]
        annotated = frame.copy()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # バウンディングボックス内の深度の中央値
                depth_roi = depth_map_norm[y1:y2, x1:x2]
                if depth_roi.size == 0:
                    continue
                z_rel = np.median(depth_roi)
                z = z_rel * scale_factor  # メートルに変換

                # ピクセル座標 → カメラ座標
                px, py = (x1 + x2) // 2, (y1 + y2) // 2
                Xc = (px - cx) * z / fx
                Yc = (py - cy) * z / fy
                Zc = z

                # 表示処理
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"XYZ: ({Xc:.2f}, {Yc:.2f}, {Zc:.2f}) m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # コンソール出力
                print(f"Person bbox: ({x1},{y1})-({x2},{y2})")
                print(f"Depth (Z): {Zc:.3f} m")
                print(f"Camera Coords (X, Y, Z): ({Xc:.3f}, {Yc:.3f}, {Zc:.3f})\n")

            cv2.imshow("center",frame)
                
            if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    else:
        break    

image_path.release()
cv2.destroyWindow()       
                
frame.save(filename="result.avi")