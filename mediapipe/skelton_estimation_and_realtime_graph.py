#x,y方向の成分のみを使用した足首と膝裏の角度の計測


import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import matplotlib.pyplot as plt




#ベクトルを２つ入れて、角度を計算する
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


try:
    # Mediapipeの初期化
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    #サッカーの動画
    path = r"C:\Users\intern01\Documents\GitHub\intern_team1\uec-internship-2026-1st-team1\video2.mp4"

    # 動画キャプチャの初期化
    # cap = cv2.VideoCapture(0)  # ここではwebカメラ
    cap = cv2.VideoCapture(path)  # ここでは保存した動画

    if not cap.isOpened():
        print("Error: カメラまたは動画を開けませんでした。")
        exit()

    # ウィンドウサイズを変更するスケール（例: 0.5 で半分の大きさ）
    resize_scale = 1.5

    # 保存する動画の設定
    output_filename = "output_pose_video.avi"
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_scale)  # 縮小後の幅
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_scale)  # 縮小後の高さ
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 動画のコーデック

    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))


    #maxlen個の角度を保存する
    #追跡やグラフ表示に使用する
    first_frame_num = 0
    last_frame_num = 50

    right_ankle_angles_traj = deque([0] * last_frame_num, maxlen=last_frame_num)
    right_knee_angles_traj = deque([0] * last_frame_num, maxlen=last_frame_num)
    left_ankle_angles_traj = deque([0] * last_frame_num, maxlen=last_frame_num)
    left_knee_angles_traj = deque([0] * last_frame_num, maxlen=last_frame_num)


    print("traj")
    print(right_ankle_angles_traj)

    #グラフの初期化
    #x軸、時間軸、1フレームごと
    graph_x = np.arange(first_frame_num, last_frame_num)

    #最初の表示部分
    lines_r_ankle, = plt.plot(graph_x, right_ankle_angles_traj, color="g", label="Angle R Ankle")
    lines_r_knee, = plt.plot(graph_x, right_knee_angles_traj, color="b", label="Angle R Knee")

    plt.xlabel("frame number")
    plt.ylabel("angle[degrees]")

    plt.xlim(first_frame_num, last_frame_num)
    plt.ylim(0, 180)

    plt.legend(loc="lower left")

    # plt.show()



    #計算したすべての関節角度を保存するリスト
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


        #取得したい関節の番号
        right_number = [mp_holistic.PoseLandmark.RIGHT_HIP, mp_holistic.PoseLandmark.RIGHT_KNEE, mp_holistic.PoseLandmark.RIGHT_ANKLE, mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX]
        left_number = [mp_holistic.PoseLandmark.LEFT_HIP, mp_holistic.PoseLandmark.LEFT_KNEE, mp_holistic.PoseLandmark.LEFT_ANKLE, mp_holistic.PoseLandmark.LEFT_FOOT_INDEX]

    

        if result.pose_landmarks:

            # print("result1")
            # print(result.pose_landmarks)
            # print(type(result.pose_landmarks))
            # print("result2")
            # print(result.pose_landmarks.landmark)
            # print(type(result.pose_landmarks.landmark))

            # print("result3 nose")
            # print(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE])
            # print(type(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]))


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
                pixel = np.array([x, y])
                # point = np.array([right.x, right.y, right.z])
                point = np.array([right.x, right.y])

                #右足
                #26 膝
                #28 足首
                #30 かかと
                #32 つま先（人差し指）
                right_detected[i] = point
                
                print("right_detedted")
                print(right_detected)



                print(f"right関節 {i}: x={x:.2f}, y={y:.2f}, z={z:.2f}")



            left_pixcel = {}
            for i in left_number:
                left = result.pose_landmarks.landmark[i]
                x = left.x * small_width    # x座標をピクセル単位に変換
                y = left.y * small_height   # y座標をピクセル単位に変換
                z = left.z            # z座標（深度情報）は正規化されている

                pixel = np.array([x, y])
                # point = np.array([left.x, left.y, left.z])
                point = np.array([left.x, left.y])

                left_pixcel[i] = [x, y, z]

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

            vec_right_ankle2knee = right_detected[26] - right_detected[28]
            vec_right_ankle2footindex = right_detected[32] - right_detected[28]
            vec_right_knee2hip = right_detected[24] - right_detected[26]
            vec_right_knee2ankle = right_detected[28] - right_detected[26]


            vec_left_ankle2knee = left_detected[25] - left_detected[27]
            vec_left_ankle2footindex = left_detected[31] - left_detected[27]
            vec_left_knee2hip = left_detected[23] - left_detected[25]
            vec_left_knee2ankle = left_detected[27] - left_detected[25]



            right_angle_ankle = calculate_angle_between_vector(vec_right_ankle2knee, vec_right_ankle2footindex)
            right_angle_knee = calculate_angle_between_vector(vec_right_knee2ankle,vec_right_knee2hip)
            left_angle_ankle = calculate_angle_between_vector(vec_left_ankle2knee, vec_left_ankle2footindex)
            left_angle_knee = calculate_angle_between_vector(vec_left_knee2hip, vec_left_knee2ankle)


            print("angle")
            print(right_angle_ankle)

            right_ankle_angles_list.append(right_angle_ankle)
            right_knee_angles_list.append(right_angle_knee)
            left_ankle_angles_list.append(left_angle_ankle)
            left_knee_angles_list.append(left_angle_knee)

            



            print("right_pixcel")
            print(right_pixcel)

            text = "x: " + str(int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_ANKLE][0])) + ", " + "y: " + str(int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_ANKLE][1])) + ", z: " + str(int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_ANKLE][2]))
            text2 = "x: " + str(int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_KNEE][0])) + ", " + "y: " + str(int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_KNEE][1])) + ", z: " + str(int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_KNEE][2]))

            # cv2.putText(small_frame,
            #     text2,
            #     org=(0, 50),
            #     fontFace=cv2.FONT_HERSHEY_DUPLEX,
            #     fontScale=1.5,
            #     color=(0, 255, 0),
            #     thickness=2,
            #     lineType=cv2.LINE_AA)


            # cv2.putText(small_frame,
            #     text,
            #     org=(0, 100),
            #     fontFace=cv2.FONT_HERSHEY_DUPLEX,
            #     fontScale=1.5,
            #     color=(0, 255, 0),
            #     thickness=2,
            #     lineType=cv2.LINE_AA)

            cv2.putText(small_frame,
                f"Angle R Ankle: {right_angle_ankle:.1f}",
                org=(0, 30),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.8,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA)


            cv2.putText(small_frame,
                f"Angle R Knee: {right_angle_knee:.1f}",
                org=(0, 60),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.8,
                color=(255, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA)

            cv2.putText(small_frame,
                f"Angle L Ankle: {left_angle_ankle:.1f}",
                org=(0, 90),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.8,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA)

            cv2.putText(small_frame,
                f"Angle L Knee: {left_angle_knee:.1f}",
                org=(0, 120),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.8,
                color=(255, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA)
            


            cv2.circle(small_frame, (int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_ANKLE][0]), int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_ANKLE][1])), 5, (0, 255, 0), -1)
            cv2.circle(small_frame, (int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_KNEE][0]), int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_KNEE][1])), 5, (255, 0, 0), -1)
            cv2.circle(small_frame, (int(left_pixcel[mp_holistic.PoseLandmark.LEFT_ANKLE][0]), int(left_pixcel[mp_holistic.PoseLandmark.LEFT_ANKLE][1])), 5, (0, 0, 255), -1)
            cv2.circle(small_frame, (int(left_pixcel[mp_holistic.PoseLandmark.LEFT_KNEE][0]), int(left_pixcel[mp_holistic.PoseLandmark.LEFT_KNEE][1])), 5, (255, 255, 0), -1)


            #角度部分をわかりやすく表示
            # cv2.ellipse(small_frame, (int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_ANKLE][0]), int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_ANKLE][1])), (50, 50), 0, 0, right_angle_knee, (0, 255, 0), 2)


            #グラフ表示のための更新
            right_ankle_angles_traj.append(right_angle_ankle)
            right_knee_angles_traj.append(right_angle_knee)
            left_ankle_angles_traj.append(left_angle_ankle)
            left_knee_angles_traj.append(left_angle_knee)

            # x += 1
            print("x")
            print(graph_x)
            first_frame_num += 1
            last_frame_num += 1
        
            lines_r_ankle.set_data(graph_x, right_ankle_angles_traj)
            lines_r_knee.set_data(graph_x, right_knee_angles_traj)
        
        
            plt.xlim(graph_x.min(), graph_x.max())
        
        
            plt.pause(0.01)


        # 縮小されたフレームを保存
        out.write(small_frame)

        # 縮小されたフレームを表示
        cv2.imshow('Pose Detection', small_frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




finally:
    print("angles_list")
    print(right_ankle_angles_list)
    print(right_knee_angles_list)
    # print(left_ankle_angles_list)
    # print(left_knee_angles_list)

    # リソースを解放
    cap.release()
    out.release()  # 保存用のVideoWriterを解放
    cv2.destroyAllWindows()
    print(f"保存された動画ファイル: {output_filename}")

