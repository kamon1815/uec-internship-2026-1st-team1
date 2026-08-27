#x,y方向の成分のみを使用した足首と膝裏の角度の計測
#リアルタイムでグラフに表示する
#クラスに分けて、わかりやすく


import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from pathlib import Path
from pypuclib import CameraFactory, Camera, XferData, Resolution, Decoder,GPUSetup

from Lifting_assistance.ball_class import BallDetecter, BallPositionTracker, ContactCounter, PoseDetecter





#骨格分析のクラス
# mediapipeで骨格推定し、関節角度などを計算する
#外部からの操作は基本的にanalyzeメソッドを使う
#基本的に１フレームずついれて処理を行うクラス、映像のフレームへの分解はクラスの外で行う
#ただし、今まで計算した角度情報などはリストやdequeとして保持する
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

    NOSE = 0



    #インストラクタ
    #mediapipeの初期化など
    def __init__(self, first_frame_num, last_frame_num):

        # Mediapipeの初期化
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(model_complexity=0,
                static_image_mode=False,
                smooth_landmarks=False,
                min_tracking_confidence=0.3)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic


        #maxlen個の角度を保存する、超えたら古いものから順に削除
        #追跡やグラフ表示に使用する
        self.first_frame_num = first_frame_num
        self.last_frame_num = last_frame_num

        self.draw_frame_num = last_frame_num - first_frame_num
    
        self.right_ankle_angles_traj = deque([0] * self.draw_frame_num, maxlen=self.draw_frame_num)
        self.right_knee_angles_traj = deque([0] * self.draw_frame_num, maxlen=self.draw_frame_num)
        self.left_ankle_angles_traj = deque([0] * self.draw_frame_num, maxlen=self.draw_frame_num)
        self.left_knee_angles_traj = deque([0] * self.draw_frame_num, maxlen=self.draw_frame_num)


        #計算したすべての関節角度を保存するリスト
        self.right_ankle_angles_list = []
        self.right_knee_angles_list = []
        self.left_ankle_angles_list = []
        self.left_knee_angles_list = []

        #取得したい関節の番号
        self.joint_num_list = [self.RIGHT_HIP, self.RIGHT_KNEE, self.RIGHT_ANKLE, self.RIGHT_FOOT_INDEX,
                        self.LEFT_HIP, self.LEFT_KNEE, self.LEFT_ANKLE, self.LEFT_FOOT_INDEX, self.NOSE]



        
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
        
        return None
    
                




    #すべての関節の中から必要な関節を取り出す
    #入力：取得したい関節の数値が入ったリスト
    #出力：必要な関節のみのx,y座標を格納した辞書（ピクセル版と元のデータ版）
    def get_selectecd_landmarks(self, frame, joint_nummbers, is_draw):
        height, width = frame.shape

        result_pose = self.detect_pose(frame, is_draw)

        # 各関節の座標を取得して出力
        #ピクセル値の保存
        joint_pixcels = {}
        #もとのデータである0~1の数値を保存
        joint_data = {}
        if result_pose is not None:
            for joint_num in joint_nummbers:
                joint = result_pose.landmark[joint_num]

                #x,yはピクセル値として保存
                x = int(joint.x * width)    # x座標をピクセル単位に変換
                y = int(joint.y * height)   # y座標をピクセル単位に変換
                z = joint.z            # z座標（深度情報）は正規化されている

                
                pixcel = np.array([x, y])
                data = np.array([joint.x, joint.y])

                joint_pixcels[joint_num] = pixcel
                joint_data[joint_num] = data

            return joint_pixcels, joint_data
        else:
            return None, None

        




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

    #角度を取得したい関節名（番号）を入れると、その関節角度が返る
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


    #基本的に外部からはこのanalyzeメソッドを使用
    def analyze(self, frame):
        #選んだ番号の関数の座標を取得
        joint_pixcels, joint_data = self.get_selectecd_landmarks(frame, self.joint_num_list, False)
        # l_joint_pixcels, l_joint_data = self.get_selectecd_landmarks(frame, self.left_joint_num, False)
        if joint_pixcels is not None:
            #関節角度の計算
            angle_r_ankle = self.calc_joint_angles(self.RIGHT_ANKLE, joint_pixcels)
            angle_r_knee = self.calc_joint_angles(self.RIGHT_KNEE, joint_pixcels)
            angle_l_ankle = self.calc_joint_angles(self.LEFT_ANKLE, joint_pixcels)
            angle_l_knee = self.calc_joint_angles(self.LEFT_KNEE, joint_pixcels)


            #リストに新たに計算した角度を追加する
            self.right_ankle_angles_list.append(angle_r_ankle)
            self.right_knee_angles_list.append(angle_r_knee)
            self.left_ankle_angles_list.append(angle_l_ankle)
            self.left_knee_angles_list.append(angle_l_knee)

            self.right_ankle_angles_traj.append(angle_r_ankle)
            self.right_knee_angles_traj.append(angle_r_knee)
            self.left_ankle_angles_traj.append(angle_l_ankle)
            self.left_knee_angles_traj.append(angle_l_knee)

            self.first_frame_num += 1
            self.last_frame_num += 1

            nose_y = joint_pixcels[self.NOSE][1]
            toe_position = [joint_pixcels[self.RIGHT_FOOT_INDEX], joint_pixcels[self.LEFT_FOOT_INDEX]]



            return {
                "right_ankle_angles_list": self.right_ankle_angles_list,
                "right_knee_angles_list": self.right_knee_angles_list,
                "left_ankle_angles_list": self.left_ankle_angles_list,
                "left_knee_angles_list": self.left_knee_angles_list,
                "right_ankle_angles_traj": self.right_ankle_angles_traj,
                "right_knee_angles_traj": self.right_knee_angles_traj,
                "left_ankle_angles_traj": self.left_ankle_angles_traj,
                "left_knee_angles_traj": self.left_knee_angles_traj,
                "first_frame_num": self.first_frame_num,
                "last_frame_num": self.last_frame_num,
                "joint_pixcels": joint_pixcels
            }, nose_y, toe_position
        else:
            return None, None, None



            
            

    # #デストラクタ
    # #処理後の解放など
    # def __del__(self):







#＋＋＋外側でやる処理（ここではmain関数、統合後にはLifftingSupportSystemクラスで行う）＋＋＋
# カメラ映像取得
# 映像表示（画面上にマークを付ける場合はPoseAnalyzerから必要な値だけをもらって外で表示するプログラムは書く）
# 映像の保存
#グラフ表示（必要な値はPoseAnalyzerから取得）

def main():
    #****************初期化設定****************
    #動画の読み込み（カメラ映像取得の代わりに）
    # path = r"C:\Users\intern01\Documents\GitHub\intern_team1\uec-internship-2026-1st-team1\video2.mp4"
    # path = r"C:\Users\intern01\Documents\GitHub\intern_team1\uec-internship-2026-1st-team1\movie\zikkenn4.avi" 
    
    
    
    BASE_DIR = Path(__file__).resolve().parent
    # path = BASE_DIR / "../movie/zikken3.avi"
    #保存した映像の読み込み
    # cam = cv2.VideoCapture(path)

    #ハイスピードカメラの設定＋＋＋＋＋＋＋＋＋
    cam = CameraFactory().create()

    #フレームレート調節
    cam.setFramerateShutter(240,240)

    # To decode image, get decoder obj from camera
    decoder = cam.decoder()

    # If a GPU device is available, decoding is done on the GPU.
    # To setup GPU device
    reso = cam.resolution()
    GPUStatus = decoder.getAvailableGPUProcess()

    if GPUStatus == True:
        param = GPUSetup(reso.width, reso.height)
        decoder.setupGPUDecode(param)
        print("Decode using a GPU device")
    elif GPUStatus == False:
        print("Since GPU is not available, decode using CPU")

    # if not cam.isOpened():
    #     print("Error: カメラまたは動画を開けませんでした。")
    #     exit()

    # Set filepath to save image
    # savePath = BASE_DIR / "hello_world.bmp"

    # Function : Save single image as BMP 
    # def saveBMP(img):
    #     cv2.imwrite(savePath, img)
    #     print("saved a BMP image")

    

    #リサイズの大きさも統合するためにクラスの外で行う
    resize_scale = 0.7

    # # 保存する動画の設定
    # output_filename = "output_pose_video.avi"
    # fps = int(cam.get(cv2.CAP_PROP_FPS)) if cam.get(cv2.CAP_PROP_FPS) > 0 else 30
    # frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_scale)  # 縮小後の幅
    # frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_scale)  # 縮小後の高さ
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 動画のコーデック

    # out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))


    #グラフ表示の際のx軸範囲初期設定（何フレーム分の角度を一度に表示したいか）＝＝＝＝＝＝＝＝＝
    #表示するフレーム数
    draw_frame_num = 50

    #x軸の範囲設定
    first_frame_num = -draw_frame_num
    last_frame_num = 0   #実際のフレーム番号はlast_frame_num - 1

    #グラフの初期化
    #x軸の幅を設定、時間軸、1フレームごと
    graph_x = np.arange(first_frame_num, last_frame_num)
    graph_y = np.zeros(draw_frame_num)

    
    #最初の表示部分
    lines_r_ankle, = plt.plot(graph_x, graph_y, color="#ff6347", label="Angle R Ankle")
    lines_r_knee, = plt.plot(graph_x, graph_y, color="#ffa500", label="Angle R Knee")
    lines_l_ankle, = plt.plot(graph_x, graph_y, color="#40e0d0", label="Angle L Ankle")
    lines_l_knee, = plt.plot(graph_x, graph_y, color="#90ee90", label="Angle L Knee")


    plt.xlabel("frame number")
    plt.ylabel("angle[degrees]")

    plt.xlim(first_frame_num, last_frame_num + 1)
    plt.ylim(0, 180)

    plt.legend(loc="lower left")

    # plt.show()

    #骨格検出するクラスのインスタンス生成
    skelton_est = PoseAnalyzer(first_frame_num, last_frame_num)


    #接触時にグラフにマーカーを描きたい
    # ボール検出のクラスの初期化
    ball_detecter = BallDetecter()
    pose_detecter = PoseDetecter()
    contact_counter = ContactCounter(contact_distance=100)
    ball_tracker = BallPositionTracker(max_missing_frame=5)
    contact_xlist = []


    while True:
        # ret, frame = cap.read()

        # if not ret:
        #     print("Error: フレームを取得できませんでした。")
        #     break

        xferData = cam.grab()

        # Decode the data can be used as image
        if GPUStatus == True:
            frame = decoder.decodeGPU(xferData, True, reso.width)
        elif GPUStatus == False:
            frame = decoder.decode(xferData)

        # Show the image
        # cv2.imshow("INFINICAM", frame)
        # フレームの高さと幅を取得
        height, width = frame.shape

        # フレームサイズを縮小
        small_frame = cv2.resize(frame, (int(width * resize_scale), int(height * resize_scale)))
        small_height, small_width = small_frame.shape


        print("small_frame_shape")
        print(small_height, small_width)


        #PoseAnalyzerのanalyzeメソッドを実行することで必要な情報が返る
        pose_result, nose_y, toe_positions = skelton_est.analyze(small_frame)
        # nose_y = pose_result["nose_y"]
        # toe_positions = pose_result["toe_position"]

        print("skelton_nose")
        print(toe_positions)
        print(nose_y)

        if pose_result is not None:



            #ボールの接触判定＋＋＋＋＋＋＋＋
            detected_ball_position,ball_box = ball_detecter.detect(small_frame)
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
            # toe_positions, nose_y = pose_detecter.detect(small_frame)

            # print("ball_nose")
            # print(toe_positions)
            # print(nose_y)
        
            #接触判定を実施
            contact,distance = contact_counter.update(detected_ball_position,toe_positions)

            print(contact)

            #＋＋＋＋＋＋＋＋＋＋＋＋＋


            #画面上に角度情報を表示
            #PoseAnalyzerに表示をいれたバージョン
            # skelton_est.display_angles(small_frame, "Angle R Ankle", (0, 30), (71, 99, 255))
            # skelton_est.display_angles(small_frame, "Angle R Knee", (0, 60), (0, 165, 255))
            # skelton_est.display_angles(small_frame, "Angle L Ankle", (0, 90), (208, 224, 64))
            # skelton_est.display_angles(small_frame, "Angle L Knee", (0, 120), (144, 238, 144))

            cv2.putText(small_frame,
                f"Angle R Ankle: {int(pose_result["right_ankle_angles_list"][-1])}",
                org=(0, 30),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.8,
                color=(71, 99, 255),
                thickness=2,
                lineType=cv2.LINE_AA)


            cv2.putText(small_frame,
                f"Angle R Knee: {int(pose_result["right_knee_angles_list"][-1])}",
                org=(0, 60),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.8,
                color=(0, 165, 255),
                thickness=2,
                lineType=cv2.LINE_AA)

            cv2.putText(small_frame,
                f"Angle L Ankle: {int(pose_result["left_ankle_angles_list"][-1])}",
                org=(0, 90),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.8,
                color=(208, 224, 64),
                thickness=2,
                lineType=cv2.LINE_AA)

            cv2.putText(small_frame,
                f"Angle L Knee: {int(pose_result["left_knee_angles_list"][-1])}",
                org=(0, 120),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.8,
                color=(144, 238, 144),
                thickness=2,
                lineType=cv2.LINE_AA)

            


            cv2.circle(small_frame, (int(pose_result["joint_pixcels"][PoseAnalyzer.RIGHT_ANKLE][0]), int(pose_result["joint_pixcels"][PoseAnalyzer.RIGHT_ANKLE][1])), 5, (71, 99, 255), -1)
            cv2.circle(small_frame, (int(pose_result["joint_pixcels"][PoseAnalyzer.RIGHT_KNEE][0]), int(pose_result["joint_pixcels"][PoseAnalyzer.RIGHT_KNEE][1])), 5, (0, 165, 255), -1)
            cv2.circle(small_frame, (int(pose_result["joint_pixcels"][PoseAnalyzer.LEFT_ANKLE][0]), int(pose_result["joint_pixcels"][PoseAnalyzer.LEFT_ANKLE][1])), 5, (208, 224, 64), -1)
            cv2.circle(small_frame, (int(pose_result["joint_pixcels"][PoseAnalyzer.LEFT_KNEE][0]), int(pose_result["joint_pixcels"][PoseAnalyzer.LEFT_KNEE][1])), 5, (144, 238, 144), -1)





            
            ######グラフ表示のための更新


            if contact:
                contact_xlist.append(pose_result["last_frame_num"] - 1)
            print("frame_num")
            print(pose_result["first_frame_num"], pose_result["last_frame_num"])
            
            graph_x = np.arange(pose_result["first_frame_num"], pose_result["last_frame_num"])
        
            
            # print(len(graph_x), len(pose_result["right_ankle_angles_traj"]))
            lines_r_ankle.set_data(graph_x, pose_result["right_ankle_angles_traj"])
            lines_r_knee.set_data(graph_x, pose_result["right_knee_angles_traj"])
            lines_l_ankle.set_data(graph_x, pose_result["left_ankle_angles_traj"])
            lines_l_knee.set_data(graph_x, pose_result["left_knee_angles_traj"])

            plt.vlines(contact_xlist, 0, 180, color="palevioletred")
            plt.xlim(graph_x.min(), graph_x.max() + 1)

            # plt.show()
        
        
            plt.pause(0.01)


            # 縮小されたフレームを保存
            # out.write(small_frame)

            # 縮小されたフレームを表示
            cv2.imshow('Pose Detection', small_frame)







    # リソースを解放
    cam.release()
    out.release()  # 保存用のVideoWriterを解放
    cv2.destroyAllWindows()
    print(f"保存された動画ファイル: {output_filename}")


if __name__ == "__main__":
    main()