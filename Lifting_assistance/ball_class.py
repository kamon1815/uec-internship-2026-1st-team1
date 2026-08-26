import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import math
import mediapipe as mp

BASE_DIR = Path(__file__).resolve().parent
path = BASE_DIR / "../movie/zikken3.avi"

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
                      
class PoseDetecter:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        

    def detect_toes(self,frame):
         frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
         result = self.pose.process(frame_rgb)

         toe_positions = []

         if result.pose_landmarks:
              height,width, _ = frame.shape
              toe_numbers = [
                   self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
                   self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX
              ]

              for toe_number in toe_numbers:
                   landmark = result.pose_landmarks.landmark[toe_number.value]

                   toe_x = int(landmark.x * width)
                   toe_y = int(landmark.y * height)

                   toe_positions.append((toe_x,toe_y))

              self.mp_drawing.draw_landmarks(
                   frame,
                   result.pose_landmarks,
                   self.mp_pose.POSE_CONNECTIONS
              )

         return toe_positions

    def nose_height(self,frame):
        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result = self.pose.process(frame_rgb)

        if result.pose_landmarks:
            height,width, _ = frame.shape
            landmark = result.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]

            nose_potision = int(landmark.y * height)

            self.mp_drawing_landmarks(
                frame,
                result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        return nose_potision
    
    def close(self):
         self.pose.close()

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

class BallHeightDetecter:
    def __init__(self):
        pass

    def update(self,ball_position, nose_height):
        if ball_position is not None:
            ball_x,ball_y = ball_position

            if nose_height - ball_y > 0:
                return True

        return False

    
