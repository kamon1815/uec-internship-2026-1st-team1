import cv2
import torch
import numpy as np
from ultralytics import YOLO


model = YOLO("yolov8n.pt") 
video_path = "C:\\Users\\intern02\\Desktop\\intern\\sample\\img\\ball.mp4"
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
#results = model("C:\\Users\\intern02\\Desktop\\intern\\sample\\img\\ball.mp4")


# MiDaSで深度推定モデル読み込み
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to("cpu").eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform


# --- キャリブレーション仮定値 ---
scale_factor = 3.0  # 相対深度をメートル換算



#for r in results:
    #boxes = r.boxes
    #for box in boxes:
        #cls_id = int(box.cls[0])
        #conf = float(box.conf[0])
        #if model.names[cls_id] == "sports ball":
            #x1,y1,x2,y2 = box.xyxy[0].tolist()
            #cx = (x1+x2)/2
            #cy = (y1+y2)/2
        #print(f"Detected:{model.names[cls_id]} with confidence{conf:.2f}")
            #print(f"中心座標:({cx},{cy})")




while cap.isOpened():
    success,frame = cap.read()
    if success:
        results = model(frame)

        imageWidth = results[0].orig_shape[0]
        imageHeight = results[0].orig_shape[1]

        names = results[0].names
        classes = results[0].boxes.cls
        boxes = results[0].boxes
        boxedframe = results[0].plot()

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = midas_transforms(img_rgb)
        with torch.no_grad():
            prediction = midas(input_tensor)
            depth_map = prediction.squeeze().cpu().numpy()
            depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
        
                # 深度マップの正規化
        depth_min, depth_max = depth_map.min(), depth_map.max()
        depth_map_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)

        results = model(frame)[0]
        annotated = frame.copy()

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if model.names[cls_id] == "person":#人
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1+x2)/2
                        cy = (y1+y2)/2
                        depth_roi = depth_map_norm[y1:y2, x1:x2]
                        if depth_roi.size == 0:
                            continue
                        z_rel_p = np.median(depth_roi)
            if model.names[cls_id] == "cup":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1+x2)/2
                        cy = (y1+y2)/2
                        depth_roi = depth_map_norm[y1:y2, x1:x2]
                        if depth_roi.size == 0:
                            continue
                        z_rel_c = np.median(depth_roi)
                        print(f"中心座標:({cx},{cy},{z_rel_p - z_rel_c})")
                        cv2.circle(boxedframe, (int(cx),int(cy)), 10,(0,0,255), -1)

            cv2.imshow("center",boxedframe) 

            if cv2.waitKey(1) & 0xFF == ord("q"):
                  break
    else:
          break    

cap.release()
cv2.destroyWindow()       

frame.save(filename="result.avi")