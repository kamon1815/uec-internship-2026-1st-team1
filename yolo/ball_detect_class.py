import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import math
import mediapipe as mp

BASE_DIR = Path(__file__).resolve().parent
path = BASE_DIR / "../movie/zikken3.avi"
model = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

class YOLO:

    def __init__(self,model,frame):
        self.frame = frame
        self.model = model

    def detect_ball(frame):
        results = model(frame)
        boxes = results[0].boxes
        
        for box in boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] == "sports ball":
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                cx = (x1+x2)/2
                cy = (y1+y2)/2
                #print(f"中心座標:({cx},{cy})")
                cv2.rectangle(frame, (int(x1), int(y1) ), (int(x2), int(y2)), (0,0,255), 2)
                cv2.circle(frame, (int(cx),int(cy)), 5,(0,0,255), -1)

    def contact_judgement(detect_ball_frame,mediapipe_frame,ball,):
        foot_numbers = [ mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX, mp_holistic.PoseLandmark.LEFT_FOOT_INDEX]
        ball_x,ball_y = ball
        for foot_number in foot_numbers:
            #つま先をピクセル座標に変換
            landmark = mediapipe_frame.pose_landmarks.landmark[foot_number]
         
            foot_x = int(landmark.x * small_width)
            foot_y = int(landmark.y * small_height)
         
            cv2.circle(frame,(foot_x,foot_y),7,(255,0,0),-1)
            #ボールとつま先の距離を計算
            distance = math.sqrt((ball_x - foot_x)**2 + (ball_y - foot_y)**2)
            #左右のつま先の近いほうを採用
            nearest_distance = min(nearest_distance,distance)
         
        if nearest_distance < CONTACT_DISTANCE:
            contact = True
         
        #現在のフレームで触れ、前のフレームで触れていないとき
        if contact and not was_contacting:
            contact_count+=1
         
        was_contacting = contact
         
        #接触中
        if contact:
            #接触した瞬間、contactと表示
            cv2.putText(frame,
                        "contact",
                        org=(30,60),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1.5,
                        color=(0,255,0),
                        thickness=2,
                        lineType=cv2.LINE_AA)
        #常に接触回数を表示     
        cv2.putText(frame,
                    f"contact count:{contact_count}",
                    org=(30,110),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.7,
                    color=(0,255,0),
                    thickness=2,
                    lineType=cv2.LINE_AA)             
         

class mediapipe:
    def __init__(self):
        right_ankle_angles_list = []
        right_knee_angles_list = []
        left_ankle_angles_list = []
        left_knee_angles_list = []
        

    def calculate_angle_between_vector(v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)

        # 単位ベクトル化
        v1_unit = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_unit = v2 / (np.linalg.norm(v2) + 1e-6)
        
        # 内積と外積
        dot = np.dot(v1_unit, v2_unit)
        cross = v1_unit[0] * v2_unit[1] - v1_unit[1] * v2_unit[0]  # 2D外積（z成分のみ）

        # atan2を使えば符号付きで角度が出せる！（-π〜+π）
        angle_rad = np.arctan2(cross, dot)
        angle_deg = np.degrees(angle_rad)
        
        # 常に 0〜360度に変換
        if angle_deg < 0:
            angle_deg += 360

        #足首などの角度は基本180度を超えることはないので、180以下へと変更する
        if angle_deg > 180:
            angle_deg = 360 - angle_deg

        return angle_deg


    def skeleton_detection(frame):
        ret, frame = cap.read()
        if not ret:
                 print("Error: フレームを取得できませんでした。")
                 break
        
        # フレームの高さと幅を取得
        height, width, _ = frame.shape
        
        # フレームサイズを縮小
        small_frame = cv2.resize(frame, (int(width * resize_scale), int(height * resize_scale)))
        
        small_height, small_width, _ = small_frame.shape
        
                # print("small_size")
                # print(small_height, small_width)
        
                nearest_distance = float("inf")
                results = model(small_frame)
                boxes = results[0].boxes

    

    
