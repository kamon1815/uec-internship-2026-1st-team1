import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import math
import mediapipe as mp

BASE_DIR = Path(__file__).resolve().parent
path = BASE_DIR / "../movie/zikken3.avi"

#YOLOによってボールを検出するクラス
class BallDetecter:
    def __init__(self):
         self.model = YOLO("yolov8n.pt")
    
    def detect(self,frame):
        results = self.model(frame)
        boxes = results[0].boxes
        
        for box in boxes:
            cls_id = int(box.cls[0])
            if self.model.names[cls_id] == "sports ball":
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                ball_x = int((x1+x2)/2)
                ball_y = int(y2)
                return (ball_x,ball_y),(x1,y1,x2,y2)
                # cv2.rectangle(frame, (int(x1), int(y1) ), (int(x2), int(y2)), (0,0,255), 2)
                # cv2.circle(frame, (int(cx),int(cy)), 5,(0,0,255), -1)
        return None,None

#mediapipeでつま先の座標、鼻の座標を得るクラス                
class PoseDetecter:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode = False,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
        )
        
    def detect(self,frame):
         frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
         result = self.pose.process(frame_rgb)

         toe_positions = []
         nose_y = None

         if result.pose_landmarks is None:
             return toe_positions,nose_y
         
         height,width, _ = frame.shape
         landmarks = result.pose_landmarks.landmark

         toe_numbers = [
                   self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
                   self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX
              ]

         for toe_number in toe_numbers:
                   landmark = result.pose_landmarks.landmark[toe_number.value]

                   toe_x = int(landmark.x * width)
                   toe_y = int(landmark.y * height)

                   toe_positions.append((toe_x,toe_y))

         nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
         nose_y = int(nose.y * height)

         self.mp_drawing.draw_landmarks(
                   frame,
                   result.pose_landmarks,
                   self.mp_pose.POSE_CONNECTIONS
              )

         return toe_positions,nose_y

    def close(self):
         self.pose.close()

#接触したことを判定するクラス
class ContactCounter:
    def __init__(self,contact_distance=40):
        self.contact_distance = contact_distance
        self.contact_count = 0
        self.was_contacting = False

    def update(self, ball_position, toe_positions):
        contact = False
        nearest_distance = float("inf")

        if ball_position is not None:
            ball_x,ball_y = ball_position

            for toe_x, toe_y in toe_positions:
                distance = math.sqrt((ball_x - toe_x) ** 2 + (ball_y - toe_y) ** 2)
                nearest_distance = min(nearest_distance,distance)

            if nearest_distance < self.contact_distance:
                contact = True

        if contact and not self.was_contacting:
            self.contact_count += 1

        self.was_contacting = contact

        return contact,nearest_distance

#ボールが高さを判定するクラス
class BallHeightDetecter:
    def __init__(self):
        self.is_toohigh = False

    def update(self,ball_position, nose_height):
        self.is_toohigh = False
        if ball_position is None or nose_height is None:
            return False
        if ball_position is not None or nose_height is None:
            ball_x,ball_y = ball_position

            if nose_height > ball_y:
                self.is_toohigh = True

        return self.is_toohigh

#ボールの座標が得られなかった時前のフレームから補完するクラス
class BallPositionTracker:
    def __init__(self, max_missing_frame=5):
        self.last_position = None
        self.missing_frame = 0
        self.max_missing_frames = max_missing_frame

    #FalseはYOLOによる検出、Trueは前回位置による補完を表す
    def update(self, detected_position):
        #YOLOがボールを検知したとき
        if detected_position is not None:
            self.last_position = detected_position
            self.missing_frames = 0

            return detected_position, False
        #YOLOがボールを検知できなかった時
        if (
            self.last_position is not None
            and self.missing_frames < self.max_missing_frames
        ):
            self.missing_frames += 1
            #前回の位置を補完位置として返す
            return self.last_position, True

        #長時間検出できなければ位置を無効にする
        self.last_position = None
        return None, False

    
