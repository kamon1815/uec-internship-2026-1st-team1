from ball_class import BallDetecter
from ball_class import PoseDetecter
from ball_class import ContactCounter
from ball_class import BallHeightDetecter
from videoprocessor import VideoProcessor

import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
video_path = BASE_DIR / "../movie/zikken3.avi"


ball_detecter = BallDetecter()
pose_detecter = PoseDetecter()
contact_counter = ContactCounter(contact_distance=60)
ballheight_detecter = BallHeightDetecter()
video = VideoProcessor(video_path,"output.avi")

while True:
    ret, frame = video.cap.read()

    if not ret:
        break

    ball_position,ball_box = ball_detecter.detect(frame)
    toe_positions = pose_detecter.detect_toes(frame)

    contact,distance = contact_counter.update(ball_position,toe_positions)

    for toe_position in toe_positions:
        cv2.circle(frame, toe_position,7,(255,0,0),-1)
    if ball_position is not None:
        cv2.circle(frame,ball_position,7,(0,0,255),-1)
        #cv2.rectangle(frame, (ball_box), (0,0,255), 2)

    if contact:
        cv2.putText(frame,
                    "contact",
                    org=(30,60),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1.5,
                    color=(0,255,0),
                    thickness=2,
                    lineType=cv2.LINE_AA)

    cv2.putText(frame,
                f"contact count:{contact_counter.contact_count}",
                org=(30,110),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.7,
                color=(0,255,0),
                thickness=2,
                lineType=cv2.LINE_AA)

    video.out.write(frame)
    cv2.imshow("lifting assistance", frame)

    if cv2.waitKey(1) &0xFF == ord('q'):
        break

pose_detecter.close()
video.close()

print(f"接触回数：{contact_counter.contact_count}回")