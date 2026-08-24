import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import math

model = YOLO("yolov8n.pt") 
#接触と判定する距離
CONTACT_DISTANCE = 30
contact_count = 0
was_contacting = False

try:
    # Mediapipeの初期化
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    #サッカーの動画
    path = "C:\\Users\\intern02\\Desktop\\GitHub\\uec-internship-2026-1st-team1\\video2.mp4"

    # 動画キャプチャの初期化
    # cap = cv2.VideoCapture(0)  # ここではwebカメラ
    cap = cv2.VideoCapture(path)  # ここでは保存した動画

    if not cap.isOpened():
        print("Error: カメラまたは動画を開けませんでした。")
        exit()

    # ウィンドウサイズを変更するスケール（例: 0.5 で半分の大きさ）
    resize_scale = 1.0

    # 保存する動画の設定
    output_filename = "output_ball_contact.avi"
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_scale)  # 縮小後の幅
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_scale)  # 縮小後の高さ
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 動画のコーデック

    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    right_ankle_angles_list = []
    right_knee_angles_list = []
    left_ankle_angles_list = []
    left_knee_angles_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: フレームを取得できませんでした。")
            break

        # フレームの高さと幅を取得
        height, width, _ = frame.shape

        # フレームサイズを縮小
        small_frame = cv2.resize(frame, (int(width * resize_scale), int(height * resize_scale)))

        small_height, small_width, _ = small_frame.shape

        print("small_size")
        print(small_height, small_width)

        nearest_distance = float("inf")
        results = model(small_frame)
        boxes = results[0].boxes

        ball = None
        contact = False

        for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = box.conf[0]
                    if model.names[cls_id] =="sports ball":
                        x1,y1,x2,y2 = box.xyxy[0].tolist()
                        cx = (x1+x2)/2
                        cy = (y1+y2)/2
                        #ボールの検出する位置
                        ball = (int(cx),int(y2))
                        #print(f"中心座標:({cx},{cy})")
                        cv2.rectangle(small_frame, (int(x1), int(y1) ), (int(x2), int(y2)), (0,0,255), 2)
                        cv2.circle(small_frame, (int(cx),int(cy)), 5,(0,0,255), -1)
        

        # BGRからRGBに変換（Mediapipeが必要とするフォーマット）
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Mediapipeで骨格検出を実行
        result = pose.process(frame_rgb)

        #取得したい関節の番号
        right_number = [mp_holistic.PoseLandmark.RIGHT_HIP, mp_holistic.PoseLandmark.RIGHT_KNEE, mp_holistic.PoseLandmark.RIGHT_ANKLE, mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX]
        left_number = [mp_holistic.PoseLandmark.LEFT_HIP, mp_holistic.PoseLandmark.LEFT_KNEE, mp_holistic.PoseLandmark.LEFT_ANKLE, mp_holistic.PoseLandmark.LEFT_FOOT_INDEX]

        foot_numbers = [ mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX, mp_holistic.PoseLandmark.LEFT_FOOT_INDEX]

        #ボールと体が画面上にある時、ボールと足の距離を計算
        if ball is not None and result.pose_landmarks is not None:
            ball_x,ball_y = ball
            for foot_number in foot_numbers:
                  landmark = result.pose_landmarks.landmark[foot_number]

                  foot_x = int(landmark.x * small_width)
                  foot_y = int(landmark.y * small_height)

                  cv2.circle(small_frame,(foot_x,foot_y),7,(255,0,0),-1)

                  distance = math.sqrt((ball_x - foot_x)**2 + (ball_y - foot_y)**2)
                  nearest_distance = min(nearest_distance,distance)

            if nearest_distance < CONTACT_DISTANCE:
                 contact = True

        #接触したとき
        if contact and not was_contacting:
             contact_count+=1

        was_contacting = contact

        #接触中
        if contact:
             cv2.putText(small_frame,
                         "contact",
                         org=(30,60),
                         fontFace=cv2.FONT_HERSHEY_DUPLEX,
                         fontScale=1.5,
                         color=(0,255,0),
                         thickness=2,
                         lineType=cv2.LINE_AA)
             
        cv2.putText(small_frame,
                         f"contact count:{contact_count}",
                         org=(30,110),
                         fontFace=cv2.FONT_HERSHEY_DUPLEX,
                         fontScale=0.7,
                         color=(0,255,0),
                         thickness=2,
                         lineType=cv2.LINE_AA)             


        if result.pose_landmarks:

            mp_drawing.draw_landmarks(
                small_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            #発見した関節を保存
            right_detected = {}
            left_detected = {}
      
            # 各関節の座標を取得して出力
            right_pixcel = {}
            for i in right_number:
                right = result.pose_landmarks.landmark[i]

                x = right.x * small_width    # x座標をピクセル単位に変換
                y = right.y * small_height   # y座標をピクセル単位に変換
                z = right.z            # z座標（深度情報）は正規化されている

                right_pixcel[i] = [x, y, z]
                pixel = np.array([x, y, z])
                point = np.array([right.x, right.y, right.z])

                #右足
                #26 膝
                #28 足首
                #30 かかと
                #32 つま先（人差し指）
                right_detected[i] = point
                
                # print("right_detedted")
                # print(right_detected)



                # print(f"right関節 {i}: x={x:.2f}, y={y:.2f}, z={z:.2f}")



            left_pixel = {}
            for i in left_number:
                left = result.pose_landmarks.landmark[i]
                x = left.x * small_width    # x座標をピクセル単位に変換
                y = left.y * small_height   # y座標をピクセル単位に変換
                z = left.z            # z座標（深度情報）は正規化されている

                pixel = np.array([x, y, z])
                point = np.array([left.x, left.y, left.z])

                left_pixel[i] = [x, y, z]

                #左足
                #25 膝
                #27 足首
                #29 かかと
                #31 つま先（人差し指）

                left_detected[i] = point
            #     print("left_detedted")
            #     print(left_detected)

            #     print(f"left関節 {i}: x={x:.2f}, y={y:.2f}, z={z:.2f}")


            # print("right_detedtec[26]")

            # vec_right_ankle2knee = right_detected[26] - right_detected[28]
            # vec_right_ankle2footindex = right_detected[32] - right_detected[28]
            # vec_right_knee2hip = right_detected[24] - right_detected[26]
            # vec_right_knee2ankle = right_detected[28] - right_detected[26]


            # vec_left_ankle2knee = left_detected[25] - left_detected[27]
            # vec_left_ankle2footindex = left_detected[31] - left_detected[27]
            # vec_left_knee2hip = left_detected[23] - left_detected[25]
            # vec_left_knee2ankle = left_detected[27] - left_detected[25]



            # right_angle_ankle = calculate_angle_between_vector(vec_right_ankle2knee, vec_right_ankle2footindex)
            # right_angle_knee = calculate_angle_between_vector(vec_right_knee2hip, vec_right_knee2ankle)
            # left_angle_ankle = calculate_angle_between_vector(vec_left_ankle2knee, vec_left_ankle2footindex)
            # left_angle_knee = calculate_angle_between_vector(vec_left_knee2hip, vec_left_knee2ankle)


            # print("angle")
            # print(right_angle_ankle)

            # right_ankle_angles_list.append(right_angle_ankle)
            # right_knee_angles_list.append(right_angle_knee)
            # left_ankle_angles_list.append(left_angle_ankle)
            # left_knee_angles_list.append(left_angle_knee)

            # print("right_pixcel")
            # print(right_pixcel)

            # text = "x: " + str(int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_ANKLE][0])) + ", " + "y: " + str(int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_ANKLE][1])) + ", z: " + str(int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_ANKLE][2]))
            # text2 = "x: " + str(int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_KNEE][0])) + ", " + "y: " + str(int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_KNEE][1])) + ", z: " + str(int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_KNEE][2]))

            # cv2.putText(small_frame,
            #     str(int(right_angle_ankle)),
            #     org=(0, 50),
            #     fontFace=cv2.FONT_HERSHEY_DUPLEX,
            #     fontScale=1.5,
            #     color=(0, 255, 0),
            #     thickness=2,
            #     lineType=cv2.LINE_AA)


            # cv2.putText(small_frame,
            #     str(int(right_angle_knee)),
            #     org=(0, 100),
            #     fontFace=cv2.FONT_HERSHEY_DUPLEX,
            #     fontScale=1.5,
            #     color=(0, 255, 0),
            #     thickness=2,
            #     lineType=cv2.LINE_AA)


            cv2.circle(small_frame, (int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_ANKLE][0]), int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_ANKLE][1])), 5, (255, 0, 0), -1)
        # 縮小されたフレームを保存
        out.write(small_frame)

        # 縮小されたフレームを表示
        cv2.imshow('Pose Detection', small_frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
     print("エラーが発生しました:", e)
    #print(right_ankle_angles_list)
    # print(right_knee_angles_list)
    # print(left_ankle_angles_list)
    # print(left_knee_angles_list)

    # リソースを解放
finally:
     cap.release()
     out.release()  # 保存用のVideoWriterを解放
     cv2.destroyAllWindows()

print(f"保存された動画ファイル: {output_filename}")
print(f"接触回数：{contact_count}")