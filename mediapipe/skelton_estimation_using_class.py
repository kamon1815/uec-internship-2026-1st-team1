#x,y方向の成分のみを使用した足首と膝裏の角度の計測
#リアルタイムでグラフに表示する
#クラスに分けて、わかりやすく


import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


#骨格分析のクラス
# mediapipeで骨格推定し、関節角度などを計算する
#外部からの操作は基本的にanalyzeメソッドを使う
class PoseAnalyzer:
    
    #mediapipeで使用している関節の各角度の数値を設定
    RIGHT_HIP = 24
    RIGHT_KNEE = 26
    RIGHT_ANKLE = 28
    RIGHT_FOOT_INDEX = 30

    LEFT_HIP = 23
    LEFT_KNEE = 25
    LEFT_ANKLE = 27
    LEFT_FOOT_INDEX = 31

    #インストラクタ
    #mediapipeの初期化など
    def __init__(self, first_frame_num, last_frame_num):

        # Mediapipeの初期化
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic


        #maxlen個の角度を保存する、超えたら古いものから順に削除
        #追跡やグラフ表示に使用する
        self.first_frame_num = first_frame_num
        self.last_frame_num = last_frame_num
    
        self.right_ankle_angles_traj = deque([0] * self.last_frame_num, maxlen=self.last_frame_num)
        self.right_knee_angles_traj = deque([0] * self.last_frame_num, maxlen=self.last_frame_num)
        self.left_ankle_angles_traj = deque([0] * self.last_frame_num, maxlen=self.last_frame_num)
        self.left_knee_angles_traj = deque([0] * self.last_frame_num, maxlen=self.last_frame_num)



        #計算したすべての関節角度を保存するリスト
        self.right_ankle_angles_list = []
        self.right_knee_angles_list = []
        self.left_ankle_angles_list = []
        self.left_knee_angles_list = []



        
    #骨格推定を行い、検出されたすべての関節を返す
    #入力：BGR形式のフレームが一枚入る
    #出力：検出されたすべての関数のリスト
    def detect_pose(self, frame, is_draw):
        

        # BGRからRGBに変換（Mediapipeが必要とするフォーマット）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mediapipeで骨格検出を実行
        result = self.pose.process(frame_rgb)

        if result.pose_landmarks:
        
            if is_draw:
                self.mp_drawing.draw_landmarks(
                    frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )

        return result.pose_landmarks
    
                




    #すべての関節の中から必要な関節を取り出す
    #入力：取得したい関節の数値が入ったリスト
    #出力：必要な関節のみのx,y座標を格納した辞書（ピクセル版と元のデータ版）
    def get_selectecd_landmarks(self, frame, joint_nummbers):
        height, width, _ = frame.shape

        result_pose = self.detect_pose(frame, False)

        # 各関節の座標を取得して出力
        #ピクセル値の保存
        joint_pixcels = {}
        #もとのデータである0~1の数値を保存
        joint_data = {}

        for joint_num in joint_nummbers:
            joint = result_pose.landmark[joint_num]

            #x,yはピクセル値として保存
            x = joint.x * width    # x座標をピクセル単位に変換
            y = joint.y * height   # y座標をピクセル単位に変換
            z = joint.z            # z座標（深度情報）は正規化されている

            
            pixcel = np.array([x, y])
            data = np.array([joint.x, joint.y])

            joint_pixcels[joint_num] = pixcel
            joint_data[joint_num] = data

        return joint_pixcels, joint_data




    # def display_angles(self, frame, angle_name, xy, color):
    #     cv2.putText(frame,
    #     f"{angle_name}: {right_angle_ankle:.1f}",
    #     org=xy,
    #     fontFace=cv2.FONT_HERSHEY_DUPLEX,
    #     fontScale=0.8,
    #     color=color,
    #     thickness=2,
    #     lineType=cv2.LINE_AA)



    #ベクトルを２つ入れて、角度を計算する
    def calculate_angle_between_vector(self, v1, v2):
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

    #角度を取得したい関節名を入れると、その関節角度が返る
    #入力：取得したい角度のナンバー（RIGHT_ANKLE = 28なら、右足の足首の角度を計測）、関節のピクセル座標を保存する辞書
    def calc_joint_angles(self, joint_num, joint_pixcels):

        if joint_num == self.RIGHT_ANKLE:
            vec_ankle2knee = joint_pixcels[self.RIGHT_KNEE] - joint_pixcels[self.RIGHT_ANKLE]
            vec_ankle2footindex = joint_pixcels[self.RIGHT_FOOT_INDEX] - joint_pixcels[self.RIGHT_ANKLE]
            result_angle = self.calculate_angle_between_vector(vec_ankle2knee, vec_ankle2footindex)
        elif joint_num == self.RIGHT_KNEE:
            vec_knee2hip = joint_pixcels[self.RIGHT_HIP] - joint_pixcels[self.RIGHT_KNEE]
            vec_knee2ankle = joint_pixcels[self.RIGHT_ANKLE] - joint_pixcels[self.RIGHT_KNEE]
            result_angle = self.calculate_angle_between_vector(vec_knee2ankle,vec_knee2hip)
        elif joint_num == self.LEFT_ANKLE:
            vec_ankle2knee = joint_pixcels[self.LEFT_KNEE] - joint_pixcels[self.LEFT_ANKLE]
            vec_ankle2footindex = joint_pixcels[self.LEFT_FOOT_INDEX] - joint_pixcels[self.LEFT_ANKLE]
            result_angle = self.calculate_angle_between_vector(vec_ankle2knee, vec_ankle2footindex)
        elif joint_num == self.LEFT_KNEE:
            vec_knee2hip = joint_pixcels[self.LEFT_HIP] - joint_pixcels[self.LEFT_KNEE]
            vec_knee2ankle = joint_pixcels[self.LEFT_ANKLE] - joint_pixcels[self.LEFT_KNEE]
            result_angle = self.calculate_angle_between_vector(vec_knee2ankle,vec_knee2hip)

        
        return result_angle


    # #基本的に外部からはこのanalyzeメソッドを使用
    # def analyze(self):



            
            

    # #デストラクタ
    # #処理後の解放など
    # def __del__(self):








def main():

    #サッカーの動画
    # path = r"C:\Users\intern01\Documents\GitHub\intern_team1\uec-internship-2026-1st-team1\video2.mp4"

    cap = cv2.VideoCapture(path)

    resize_scale = 0.7

    # 保存する動画の設定
    output_filename = "output_pose_video.avi"
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_scale)  # 縮小後の幅
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_scale)  # 縮小後の高さ
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 動画のコーデック

    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))



    #グラフ表示の際のx軸範囲設定（何フレーム分の角度を一度に表示したいか）
    first_frame_num = 0
    last_frame_num = 50

    #骨格検出するクラスのインスタンス生成
    skelton_est = PoseAnalyzer(first_frame_num, last_frame_num)

    #グラフの初期化
    #x軸の幅を設定、時間軸、1フレームごと
    graph_x = np.arange(first_frame_num, last_frame_num)

    #最初の表示部分
    lines_r_ankle, = plt.plot(graph_x, skelton_est.right_ankle_angles_traj, color="#ff6347", label="Angle R Ankle")
    lines_r_knee, = plt.plot(graph_x, skelton_est.right_knee_angles_traj, color="#ffa500", label="Angle R Knee")
    lines_l_ankle, = plt.plot(graph_x, skelton_est.left_ankle_angles_traj, color="#40e0d0", label="Angle L Ankle")
    lines_l_knee, = plt.plot(graph_x, skelton_est.left_knee_angles_traj, color="#90ee90", label="Angle L Knee")

    plt.xlabel("frame number")
    plt.ylabel("angle[degrees]")

    plt.xlim(first_frame_num, last_frame_num)
    plt.ylim(0, 180)

    plt.legend(loc="lower left")



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

       
    

        #取得したい関節の番号
        right_joint_num = [PoseAnalyzer.RIGHT_HIP, PoseAnalyzer.RIGHT_KNEE, PoseAnalyzer.RIGHT_ANKLE, PoseAnalyzer.RIGHT_FOOT_INDEX]
        left_joint_num = [PoseAnalyzer.LEFT_HIP, PoseAnalyzer.LEFT_KNEE, PoseAnalyzer.LEFT_ANKLE, PoseAnalyzer.LEFT_FOOT_INDEX]

        #選んだ番号の関数の座標を取得
        r_joint_pixcels, r_joint_data = skelton_est.get_selectecd_landmarks(small_frame, right_joint_num)
        l_joint_pixcels, l_joint_data = skelton_est.get_selectecd_landmarks(small_frame, left_joint_num)

        #関節角度の計算
        angle_r_ankle = skelton_est.calc_joint_angles(skelton_est.RIGHT_ANKLE, r_joint_pixcels)
        angle_r_knee = skelton_est.calc_joint_angles(skelton_est.RIGHT_KNEE, r_joint_pixcels)
        angle_l_ankle = skelton_est.calc_joint_angles(skelton_est.LEFT_ANKLE, l_joint_pixcels)
        angle_l_knee = skelton_est.calc_joint_angles(skelton_est.LEFT_KNEE, l_joint_pixcels)


        #リストに新たに計算した角度を追加する
        skelton_est.right_ankle_angles_list.append(angle_r_ankle)
        skelton_est.right_knee_angles_list.append(angle_r_knee)
        skelton_est.left_ankle_angles_list.append(angle_l_ankle)
        skelton_est.left_knee_angles_list.append(angle_l_knee)


        # #画面上に角度情報を表示
        # skelton_est.display_angles(small_frame, "Angle R Ankle", (0, 30), (71, 99, 255))
        # skelton_est.display_angles(small_frame, "Angle R Knee", (0, 60), (0, 165, 255))
        # skelton_est.display_angles(small_frame, "Angle L Ankle", (0, 90), (208, 224, 64))
        # skelton_est.display_angles(small_frame, "Angle L Knee", (0, 120), (144, 238, 144))

        


        # cv2.circle(small_frame, (int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_ANKLE][0]), int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_ANKLE][1])), 5, (71, 99, 255), -1)
        # cv2.circle(small_frame, (int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_KNEE][0]), int(right_pixcel[mp_holistic.PoseLandmark.RIGHT_KNEE][1])), 5, (0, 165, 255), -1)
        # cv2.circle(small_frame, (int(left_pixcel[mp_holistic.PoseLandmark.LEFT_ANKLE][0]), int(left_pixcel[mp_holistic.PoseLandmark.LEFT_ANKLE][1])), 5, (208, 224, 64), -1)
        # cv2.circle(small_frame, (int(left_pixcel[mp_holistic.PoseLandmark.LEFT_KNEE][0]), int(left_pixcel[mp_holistic.PoseLandmark.LEFT_KNEE][1])), 5, (144, 238, 144), -1)


        
        ######グラフ表示のための更新
        skelton_est.right_ankle_angles_traj.append(angle_r_ankle)
        skelton_est.right_knee_angles_traj.append(angle_r_knee)
        skelton_est.left_ankle_angles_traj.append(angle_l_ankle)
        skelton_est.left_knee_angles_traj.append(angle_l_knee)

    
    
        # lines_r_ankle.set_data(graph_x, right_ankle_angles_traj)
        # lines_r_knee.set_data(graph_x, right_knee_angles_traj)
        # lines_l_ankle.set_data(graph_x, left_ankle_angles_traj)
        # lines_l_knee.set_data(graph_x, left_knee_angles_traj)




    
    
        plt.xlim(graph_x.min(), graph_x.max())
    
    
        plt.pause(0.01)
        ######


        # 縮小されたフレームを保存
        out.write(small_frame)

        # 縮小されたフレームを表示
        cv2.imshow('Pose Detection', small_frame)



        # リソースを解放
        cap.release()
        out.release()  # 保存用のVideoWriterを解放
        cv2.destroyAllWindows()
        print(f"保存された動画ファイル: {output_filename}")


if __name__ == "__main__":
    main()