from ball_class import BallDetecter
from ball_class import PoseDetecter
from ball_class import ContactCounter
from ball_class import BallHeightDetecter
from videoprocessor import VideoProcessor
from ball_class import BallPositionTracker

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
ball_tracker = BallPositionTracker(max_missing_frame=5)
video = VideoProcessor(video_path,"output.avi")

while True:
    ret, frame = video.cap.read()

    if not ret:
        break
    #YOLOによりボールの位置を追跡
    detected_ball_position,ball_box = ball_detecter.detect(frame)
    #ボール位置を補完
    ball_position,is_predicted = ball_tracker.update(detected_ball_position)

    if ball_position is not None:
        ball_x,ball_y = ball_position

        if is_predicted:
            color = (0,255,255)
        else:
            color = (0,0,255)

        cv2.circle(frame,(ball_x,ball_y), 7, color, -1)

        if is_predicted:
            cv2.putText(frame, "Predicted", (ball_x+10,ball_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    #mediapipeを実行して鼻とつま先の座標を取得
    toe_positions, nose_y = pose_detecter.detect(frame)

    #ボールの高さ判定
    is_toohigh = ballheight_detecter.update(ball_position,nose_y)
    if is_toohigh:
        cv2.putText(frame,"BALL TOO HIGH", (30,160), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,0,255), 3, cv2.LINE_AA)

    #接触判定を実施
    contact,distance = contact_counter.update(detected_ball_position,toe_positions)

    for toe_position in toe_positions:
        cv2.circle(frame, toe_position,7,(255,0,0),-1)
    if detected_ball_position is not None:
        cv2.circle(frame,detected_ball_position,7,(0,0,255),-1)
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