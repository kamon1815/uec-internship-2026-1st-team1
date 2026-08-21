import cv2
import mediapipe as mp
import numpy as np





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

    return angle_deg



try:
    # Mediapipeの初期化
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # 動画キャプチャの初期化
    cap = cv2.VideoCapture(0)  # ここではwebカメラ

    if not cap.isOpened():
        print("Error: カメラまたは動画を開けませんでした。")
        exit()

    # ウィンドウサイズを変更するスケール（例: 0.5 で半分の大きさ）
    resize_scale = 1

    # 保存する動画の設定
    output_filename = "output_pose_video.avi"
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_scale)  # 縮小後の幅
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_scale)  # 縮小後の高さ
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 動画のコーデック

    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))


    right_ankle_angles = []
    right_knee_angles = []
    left_ankle_angles = []
    left_knee_angles = []


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: フレームを取得できませんでした。")
            break

        # フレームの高さと幅を取得
        height, width, _ = frame.shape

        # フレームサイズを縮小
        small_frame = cv2.resize(frame, (int(width * resize_scale), int(height * resize_scale)))

        # BGRからRGBに変換（Mediapipeが必要とするフォーマット）
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Mediapipeで骨格検出を実行
        result = pose.process(frame_rgb)
    

        # 検出結果を描画
        # if result.pose_landmarks:
        #     mp_drawing.draw_landmarks(
        #         small_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
        #     )
        #     # 各関節の座標を取得して出力
        #     for i, landmark in enumerate(result.pose_landmarks.landmark):
        #         x = landmark.x * width    # x座標をピクセル単位に変換
        #         y = landmark.y * height   # y座標をピクセル単位に変換
        #         z = landmark.z            # z座標（深度情報）は正規化されている
        #         print(f"関節 {i}: x={x:.2f}, y={y:.2f}, z={z:.2f}")



        right_foot = [mp_holistic.PoseLandmark.RIGHT_KNEE, mp_holistic.PoseLandmark.RIGHT_ANKLE, mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX]
        left_foot = [mp_holistic.PoseLandmark.LEFT_KNEE, mp_holistic.PoseLandmark.LEFT_ANKLE, mp_holistic.PoseLandmark.LEFT_FOOT_INDEX]

        



        if result.pose_landmarks:
            print("result1")
            print(result.pose_landmarks)
            print(type(result.pose_landmarks))
            print("result2")
            print(result.pose_landmarks.landmark)
            print(type(result.pose_landmarks.landmark))

            print("result3 nose")
            print(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE])
            print(type(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]))


            mp_drawing.draw_landmarks(
                small_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            right_detected = {}
            left_detected = {}

            
            # 各関節の座標を取得して出力
            for i in right_foot:
                right = result.pose_landmarks.landmark[i]

                x = right.x * width    # x座標をピクセル単位に変換
                y = right.y * height   # y座標をピクセル単位に変換
                z = right.z            # z座標（深度情報）は正規化されている

                point = np.array([right.x, right.y, right.z])
                #右足
                #26 膝
                #28 足首
                #30 かかと
                #32 つま先（人差し指）
                right_detected[i] = point
                
                print("right_detedted")
                print(right_detected)



                print(f"right関節 {i}: x={x:.2f}, y={y:.2f}, z={z:.2f}")


            for i in left_foot:
                left = result.pose_landmarks.landmark[i]
                x = left.x * width    # x座標をピクセル単位に変換
                y = left.y * height   # y座標をピクセル単位に変換
                z = left.z            # z座標（深度情報）は正規化されている

                point = np.array([left.x, left.y, left.z])

                #左足
                #25 膝
                #27 足首
                #29 かかと
                #31 つま先（人差し指）

                left_detected[i] = point
                print("left_detedted")
                print(left_detected)

                print(f"left関節 {i}: x={x:.2f}, y={y:.2f}, z={z:.2f}")


            print("right_detedtec[26]")
            vec_right_26_28 = right_detected[26] - right_detected[28]
            vec_right_32_28 = right_detected[32] - right_detected[28]



            right_angle_ankle = calculate_angle_between_vector(vec_right_26_28, vec_right_32_28)
            print("angle")
            print(right_angle_ankle)

            right_angles.append(right_angle_ankle)








        # 縮小されたフレームを保存
        out.write(small_frame)

        # 縮小されたフレームを表示
        cv2.imshow('Pose Detection', small_frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




except:
    print(angles)

    # リソースを解放
    cap.release()
    out.release()  # 保存用のVideoWriterを解放
    cv2.destroyAllWindows()
    print(f"保存された動画ファイル: {output_filename}")

