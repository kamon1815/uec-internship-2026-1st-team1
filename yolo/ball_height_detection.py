import cv2
from ultralytics import YOLO
import mediapipe as mp
from pathlib import Path

model = YOLO("yolov8n.pt") 

try:
    # Mediapipeの初期化
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_holistic = mp.solutions.holistic

    #サッカーの動画
    BASE_DIR = Path(__file__).resolve().parent
    #path = "C:\\Users\\intern02\\Desktop\\GitHub\\uec-internship-2026-1st-team1\\video2.mp4"
    path = BASE_DIR / "../movie/zikken3.avi"

    # 動画キャプチャの初期化
    # cap = cv2.VideoCapture(0)  # ここではwebカメラ
    cap = cv2.VideoCapture(path)  # ここでは保存した動画

    if not cap.isOpened():
        print("Error: カメラまたは動画を開けませんでした。")
        exit()

    # ウィンドウサイズを変更するスケール（例: 0.5 で半分の大きさ）
    resize_scale = 1.0

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

        #ボール検出
        results = model(small_frame)
        boxes = results[0].boxes
        ball = None

        for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = box.conf[0]
                    if model.names[cls_id] =="sports ball":
                        x1,y1,x2,y2 = box.xyxy[0].tolist()
                        cx = (x1+x2)/2
                        cy = (y1+y2)/2
                        #ボールの検出する位置
                        ball = (int(cx),int(cy))
                        #print(f"中心座標:({cx},{cy})")
                        cv2.rectangle(small_frame, (int(x1), int(y1) ), (int(x2), int(y2)), (0,0,255), 2)
                        cv2.circle(small_frame, (int(cx),int(cy)), 5,(0,0,255), -1)
        

        # BGRからRGBに変換（Mediapipeが必要とするフォーマット）
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        # Mediapipeで骨格検出を実行
        result = pose.process(frame_rgb)

        #ボールと体が画面上にある時、ボールと足の距離を計算
        if ball is not None and result.pose_landmarks is not None:
            ball_x,ball_y = ball
            #頭のをピクセル座標に変換
            landmark = result.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
            head_x = int(landmark.x * small_width)
            head_y = int(landmark.y * small_height)
            #ボールと頭の高さを計算
            distance = head_y - ball_y
            if distance > 0:
                #頭を超えたとき、too highと表示
                cv2.putText(small_frame,
                            "TOO HIGH!!!",
                            org=(30,60),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=1.5,
                            color=(255,0,0),
                            thickness=2,
                            lineType=cv2.LINE_AA)

        # 縮小されたフレームを表示
        cv2.imshow('Pose Detection', small_frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
     print("エラーが発生しました:", e)

finally:
     cap.release()
     cv2.destroyAllWindows()