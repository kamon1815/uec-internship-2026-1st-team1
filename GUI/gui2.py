import tkinter as tk
import tkinter.font as tkfont
import cv2
from tkinter import ttk
from tkinter import filedialog
from pathlib import Path
from PIL import Image, ImageTk # need to import extra module "pip install pillow"

import numpy as np
import os, csv, json, threading
from enum import IntEnum

import math
import mediapipe as mp

import pypuclib
from pypuclib import CameraFactory, Camera, XferData, Resolution, Decoder

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from Lifting_assistance.ball_class import BallPositionTracker, BallDetecter, ContactCounter, PoseDetecter,BallHeightDetecter

from matplotlib.backends.backend_tkagg import (
	FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import matplotlib.animation as animation


from skelton_estimation.camera2 import PoseAnalyzer





BASE_DIR = Path(__file__).resolve().parent

class FILE_TYPE(IntEnum):
    CSV = 0
    BINARY = 1

class BinaryReader():
    def __init__(self, name):
        # read json file
        self.dict = dict()
        with open(name, mode='rt', encoding='utf-8') as file:
            self.dict = json.load(file)

        self.file = open(name.replace(".json", ".npy"), "rb")

        # read file info
        d = np.load(self.file)
        self.framesize = self.file.tell()
        self.file.seek(0, os.SEEK_END)
        self.filesize = self.file.tell()
        self.framecount = int(self.filesize / self.framesize)
        self.file.seek(0)

        # prepare for decode
        self.decoder = Decoder(self.dict["quantization"])
        self.width = self.dict["width"]
        self.height = self.dict["height"]

        self.opened = True

    def read(self, frameNo, raw = False):
        self.file.seek(self.framesize * frameNo)
        array = np.load(self.file)
        if raw:
            return array
        else:
            return self.decoder.decode(array, Resolution(self.width, self.height))

    def readseqNo(self,frameNo):
        self.file.seek(self.framesize * frameNo)
        array = np.load(self.file)
        return self.decoder.extractSequenceNo(array, self.width, self.height)

class FileCreator():
    def __init__(self, name, filetype):
        if filetype == FILE_TYPE.CSV:
            self.file = open(name + ".csv", 'w')
            self.writer = csv.writer(self.file, lineterminator='\n')
            self.writer.writerow(["SequenceNo", "diff"])
        elif filetype == FILE_TYPE.BINARY:
            self.file = open(name + ".npy", 'wb')
        else:
            return

        self.oldSeq = 0
        self.opened = True
        self.filetype = filetype

    def write(self, xferData):
        if self.opened:
            if self.filetype == FILE_TYPE.CSV:
                self.write_csv(xferData.sequenceNo())
            elif self.filetype == FILE_TYPE.BINARY:
                self.write_binary(xferData.sequenceNo(),
                                  xferData.data())

    def write_csv(self, seq):
        if self.oldSeq != seq:
            self.writer.writerow(
                        [str(seq), 
                        str(seq - self.oldSeq), 
                        "*" if (seq - self.oldSeq) > 1 else ""])
            self.oldSeq = seq

    def write_binary(self, seq, nparray):
        if self.oldSeq != seq:
            np.save(self.file, nparray)
            self.oldSeq = seq

    def create_json(name, cam):
        data = dict()
        data["framerate"] = cam.framerate()
        data["shutter"] = cam.shutter()
        data["width"] = cam.resolution().width
        data["height"] = cam.resolution().height
        data["quantization"] = cam.decoder().quantization()
        with open(name+".json", mode='wt', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def close(self):
        if self.opened:
            self.file.close()


class SbTextFrame(tk.Frame):
    def __init__(self,master):
        super().__init__(master)
        text = tk.Text(self,wrap='none',undo=True)
        x_sb = tk.Scrollbar(self,orient='horizontal')
        y_sb = tk.Scrollbar(self,orient='vertical')
        x_sb.config(command=text.xview)
        y_sb.config(command=text.yview)
        text.config(xscrollcommand=x_sb.set,yscrollcommand=y_sb.set)
        text.grid(column=0,row=0,sticky='nsew')
        x_sb.grid(column=0,row=1,sticky='ew')
        y_sb.grid(column=1,row=0,sticky='ns')
        self.columnconfigure(0,weight=1)
        self.rowconfigure(0,weight=1)
        self.text = text
        self.x_sb = x_sb
        self.y_sb = y_sb


def add_tab(fname):
    global tframes,fnames,notebook

    tframe=SbTextFrame(notebook)
    tframes.append(tframe)

    if os.path.isfile(fname):
        f=open(fname,'r')
        lines=f.readlines()
        f.close()
        for line in lines:
            tframe.text.insert('end',line)


    fnames.append(fname)
    title=os.path.basename(fname)

    notebook.add(tframe,text=title)


#CamApplicatonとfileApplicationがGUIの表示される部分


class CamApplication(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)
        master.title("gui")
        master.geometry("800x600")


        #ウィンドウの作成
        self.cam_window = tk.Toplevel(self)
        self.cam_window.title("camera")
        self.cam_window.geometry("800x600")
        #カメラの画面生成
        self.cam = CameraFactory().create()
        #フレームレート調節
        self.cam.setFramerateShutter(240,240)
        self.fcreator = None
        self.decoder = self.cam.decoder()
        #canvasの作成、変数名をcanvasにすると下の方で生成してるcanvasとかぶる
        self.canvas_window = tk.Canvas(self.cam_window)
        self.canvas_window.pack(fill = tk.BOTH,expand=True)


        #フレームレートで選択できる値
        self.framerateValues = [1, 10, 50, 100, 125, 250, 500, 950, 1000, 
                                        1500, 2000, 2500, 3000, 3200, 4000, 5000, 
                                        8000, 10000, 20000, 25000, 30000]

        
        self.framerateStr = tk.IntVar()
        self.resolutionStr = tk.StringVar()
        self.shutterStr = tk.StringVar()
        self.acqutionVal = tk.IntVar()
        self.savefileVal = tk.IntVar()
        self.uistopVal = tk.BooleanVar()
        self.isRec = False
        self.locker = threading.Lock()
        self.font = tkfont.Font(self,family="Arial",size=10,weight="bold")
        self.judge_height = tk.StringVar(value = "高さ判定")
        self.judgeposition = tk.StringVar(value = "姿勢判定")

        #gridを用いて画面作成
        #画面を左右に分けている
        #weightで余白ができた際の比率をきめてるいまは20:1
        self.grid_columnconfigure(0,weight=20)
        self.grid_columnconfigure(1,weight=1)
        self.grid_rowconfigure(0,weight=1)

        self.left_container = ttk.Frame(self)
        self.left_container.grid(row=0,column=0,sticky = "nsew",padx=5,pady=5)
        self.right_container = ttk.Frame(self)
        self.right_container.grid(row=0,column=1,sticky = "nsew",padx=5,pady=5)

        # self.height_judge = BallHeightDetecter()

        # self.position_judge = BallHeightDetecter()

        #PoseAnalyzerで必要なものの初期化
        self.draw_frame_num = 50

        #x軸の範囲設定
        self.first_frame_num = -self.draw_frame_num
        self.last_frame_num = 0   #実際のフレーム番号はlast_frame_num - 1



        self.setup_left_side()
        self.setup_right_side()

        self.create_graph()

        

        #骨格検出するクラスのインスタンス生成
        self.skelton_est = PoseAnalyzer(self.first_frame_num, self.last_frame_num)


        #接触時にグラフにマーカーを描きたい
        # ボール検出のクラスの初期化
        self.ball_detecter = BallDetecter()
        self.contact_counter = ContactCounter(contact_distance=150, cooldown_frames=80)
        self.ballheight_detecter = BallHeightDetecter()
        self.ball_tracker = BallPositionTracker(max_missing_frame=5)
        self.contact_xlist = []


        self.delay = 15
        self.updateID = 0
        self.update()


    #左側の作成
    def setup_left_side(self):
        #まず左側を上下に分割、余白ができた際の比率を8:2に設定
        self.left_container.grid_columnconfigure(0,weight=1)
        self.left_container.grid_rowconfigure(0,weight=8)
        self.left_container.grid_rowconfigure(1,weight=2)


        #----------------------------------------------
        #graphの生成
        #---------------------------------------------

        self.graph = ttk.LabelFrame(self.left_container,
                                    text="graph"
                                     )

        self.graph.grid(row=0,column=0,sticky="nsew",padx=5,pady=5)

        #self.graph_label = ttk.Label(self.canvas,text ="グラフの作成")
        #self.graph_label.pack(padx=10,pady=20)

        #---------------------------------------------------
        # 高さ判定のフレームの作成
        #---------------------------------------------------
        self.height_judge_Frame = ttk.LabelFrame(self.left_container,
                                                text="height"
                                                )
        self.height_judge_Frame.grid(row=1,column=0,sticky="nsew",padx=5,pady=5)

        #self.height_judge = BallHeightDetecter()

        self.height_judge_label = ttk.Label(self.height_judge_Frame, font=("Arial", 20))
        self.height_judge_label.pack(padx=10,pady=20)

    #右側の作成
    def setup_right_side(self):
        #まず上下に分割、余白の比率は2:8
        self.right_container.grid_columnconfigure(0,weight=1)
        self.right_container.grid_rowconfigure(0,weight=2)
        self.right_container.grid_rowconfigure(1,weight=8)
        #---------------------------------------------------
        # 姿勢判定の表示を行う
        #---------------------------------------------------

        self.posture_judge_frame = ttk.LabelFrame(self.right_container, 
                                          text="sisei"
                                          )
        self.posture_judge_frame.grid(row=0,column=0,sticky="nsew",padx=5,pady=5)

        self.posture_judge_label = ttk.Label(self.posture_judge_frame, font=("Arial", 20))
        self.posture_judge_label.pack(padx=10,pady=20)
        
        #---------------------------------------------------
        # 設定ボタンの配置
        #---------------------------------------------------
        self.setframe = ttk.LabelFrame(self.right_container, 
                                        text="set"
                                        )
        self.setframe.grid(row=1,column=0,sticky="nsew",padx=5,pady=5)

        self.setframe.grid_rowconfigure(0,weight=1)

        #設定用の場所を10分割

        for i in range(10):
            self.setframe.grid_rowconfigure(i,weight=2,)

        #各種ボタンを配置rowとcolumnで行列の場所を決定している

        #-----------------------------------------------------
        #framerate
        #-----------------------------------------------------
            
        self.framerateLabel = ttk.Label(self.setframe,text="Framerate[fps]", width=20)
        self.framerateLabel.grid(row=0,column=0,sticky="news",padx=5, pady=5)
            
        self.framerateList = ttk.Combobox(self.setframe, 
                                                      values=self.framerateValues, 
                                                      textvariable=self.framerateStr)
        self.framerateList.grid(row=0,column=1,sticky="news",padx=5, pady=5)
        self.framerateList.bind("<<ComboboxSelected>>", self.updateFramerate)

        #---------------------------------------------------
        # shutter
        #---------------------------------------------------
            

        self.shutterLabel = ttk.Label(self.setframe,text="Shutter speed[sec]", width=20)
        self.shutterLabel.grid(row=1,column=0,sticky="news",padx=5, pady=5)
        
        self.shutterList = ttk.Combobox(self.setframe,  
                                                textvariable=self.shutterStr)
        self.shutterList.grid(row=1,column=1,sticky="news",padx=5, pady=5)
        self.shutterList.bind("<<ComboboxSelected>>", self.updateShutter)


        #---------------------------------------------------
        # resolution
        #---------------------------------------------------

        self.resolutionLabel = ttk.Label(self.setframe,text="Resolution[pixel]", width=20)
        self.resolutionLabel.grid(row=2,column=0,sticky="news",padx=5, pady=5)
        
        self.resolutionList = ttk.Combobox(self.setframe,  
                                                   textvariable=self.resolutionStr)
        self.resolutionList.grid(row=2,column=1,sticky="news",padx=5, pady=5)
        self.resolutionList.bind("<<ComboboxSelected>>", self.updateResolution)


        #---------------------------------------------------
        # Acquisition mode
        #---------------------------------------------------

        self.acqusitionLabel = ttk.Label(self.setframe,text="Acquisition mode", width=18)
        self.acqusitionLabel.grid(row=3,column=0,sticky="news",padx=5, pady=5)
    
        self.acqusition1 = tk.Radiobutton(self.setframe,
                                              text="single",
                                              value=0,
                                              variable=self.acqutionVal,
                                              command=self.updateAcquisition)
        self.acqutsiion2 = tk.Radiobutton(self.setframe,
                                              text="continuous",
                                              value=1,
                                              variable=self.acqutionVal,
                                              command=self.updateAcquisition)
        self.acqusition1.grid(row=3,column=1,sticky="news",padx=5, pady=5)
        self.acqutsiion2.grid(row=4,column=1,sticky="news",padx=5, pady=5)


        #---------------------------------------------------
        # savefiles
        #---------------------------------------------------

        self.savefilesLabel = ttk.Label(self.setframe,text="Save file", width=18)
        self.savefilesLabel.grid(row=5,column=0,sticky="news",padx=5, pady=5)

        self.savefile_csv = tk.Radiobutton(self.setframe,
                                           text="csv",
                                           value=FILE_TYPE.CSV.value,
                                           variable=self.savefileVal)
        self.savefile_bin = tk.Radiobutton(self.setframe,
                                           text="binary",
                                           value=FILE_TYPE.BINARY.value,
                                           variable=self.savefileVal,)
        self.savefile_csv.grid(row=5,column=1,sticky="news",padx=5, pady=5)
        self.savefile_bin.grid(row=6,column=1,sticky="news",padx=5, pady=5)


        #---------------------------------------------------
        # record button
        #---------------------------------------------------

        self.recButton = ttk.Button(self.setframe, 
                                    text = "REC", 
                                    command=self.rec,
                                    width=15)
        self.recButton.grid(row=7,column=0,sticky="news",padx=5, pady=5)
        self.uistopCheck = ttk.Checkbutton(self.setframe,
                                           text="Stop Live",
                                           variable=self.uistopVal,
                                           command=self.uistop)
        self.uistopCheck.grid(row=7,column=1,sticky="news",padx=5, pady=5)

        self.resetSeqButton = ttk.Button(self.setframe,
                                         text = "Reset Seq No",
                                         command = self.resetSequenceNo,
                                         width = 30)
        self.resetSeqButton.grid(row=8,column=0,sticky="news",padx=5, pady=5)      


        #---------------------------------------------------
        # reset button
        #---------------------------------------------------
        self.resetButton = ttk.Button(self.setframe, 
                                    text = "RESET", 
                                    command=self.resetDevice,
                                    width=15)
        self.resetButton.grid(row=9,column=0,sticky="news",padx=5, pady=5)   


        #---------------------------------------------------
        # initialize ui
        #---------------------------------------------------
        self.framerateList.set(self.cam.framerate())
        self.updateResolutionList()
        self.updateShutterList()
        self.updateAcquisition()



    def draw_image(self,photo):
        self.canvas.create_image(0,0,image=photo,anchor=tk.NW)

    #グラフやカメラ映像の更新はすべてここにまとめる
    def update(self):
        #ハイスピードカメラからデータを取得
        data = self.cam.grab()
        small_frame = self.updatecanvas(data)

        pose_result, nose_y, toe_positions = self.skelton_est.analyze(small_frame)
        if pose_result is not None:

            #ボールの接触判定＋＋＋＋＋＋＋＋
            #YOLOによりボールの位置を追跡
            detected_ball_position,ball_box = self.ball_detecter.detect(small_frame)
            #ボール位置を補完
            ball_position,is_predicted = self.ball_tracker.update(detected_ball_position)
        
            if ball_position is not None:
                ball_x,ball_y = ball_position
        
                if is_predicted:
                    color = (0,255,255)
                else:
                    color = (0,0,255)
        
                # cv2.circle(frame,(ball_x,ball_y), 7, color, -1)
                r = 4
                self.canvas_window.create_oval(ball_x + self.pos[0] - r, ball_y + self.pos[1] - r, ball_x + self.pos[0] + r, ball_y + self.pos[1] + r, fill="white")

                
        
                if is_predicted:
                    # cv2.putText(frame, "Predicted", (ball_x+10,ball_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    self.canvas_window.create_text(ball_x + self.pos[0]+10,ball_y + self.pos[1], anchor="nw", 
                                                    text="Predicted",
                                                    font=self.font, fill="white")
        
            #ボールの高さ判定
            is_toohigh = self.ballheight_detecter.update(ball_position,nose_y)

            self.update_height_judge_label(is_toohigh)
            
        
            #接触判定を実施
            contact = self.contact_counter.update(detected_ball_position,toe_positions)

            print("contact")
            print(contact)
            print(detected_ball_position)
            print(toe_positions)
            self.update_contact_count_label(contact)
            

           

            #＋＋＋＋＋＋＋＋＋＋＋＋＋

          
            
            # print("detect")c
            self.update_graph(pose_result, contact)
            self.update_angle_text_and_draw(pose_result, small_frame, toe_positions)


        self.updateID = self.after(self.delay, self.update)

    def updatecanvas(self, data):
        cw = self.canvas_window.winfo_width()
        ch = self.canvas_window.winfo_height()
        w = data.resolution().width
        h = data.resolution().height
        scale = 1
        if cw > 1 and ch > 1:
            scale = cw/w if cw/w < ch/h else ch/h

        frame = self.decoder.decode(data)

        # フレームサイズを縮小
        small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        
                

        # i = Image.fromarray(frame_rgb).resize((int(w*scale), int(h*scale)))
        i = Image.fromarray(frame_rgb)


    

        #コントラストを変える
        # small_frame = cv2.convertScaleAbs(small_frame, alpha=2.0, beta=60)


        self.img = ImageTk.PhotoImage(image=i)
        
        self.pos = [(cw-i.width)/2,(ch-i.height)/2]
        self.seq_no = data.sequenceNo()

        self.update_canvaswindow()

        
        
        return small_frame

    def update_canvaswindow(self, ):
        self.canvas_window.delete("all")
        self.canvas_window.create_image(self.pos[0], self.pos[1], anchor="nw", image=self.img)
        # self.canvas_window.create_text(self.pos[0]+5, self.pos[1]+5, anchor="nw", 
        #                         text="SequeceNo:" + str(self.seq_no),
        #                         font=self.font, fill="white")


        


    def updateFramerate(self, e):
        rate = self.framerateStr.get()
        self.cam.setFramerateShutter(rate, rate)
        self.updateResolutionList()
        self.updateShutterList()

    def updateShutter(self, e):
        resStr = self.shutterStr.get().split("1/")
        self.cam.setShutter(int(resStr[1]))

    def updateResolution(self, e):
        resStr = self.resolutionStr.get().split("x")
        self.cam.setResolution(int(resStr[0]), int(resStr[1]))

    def updateResolutionList(self):
        resMax = self.cam.resolutionMax()
        resLimit = self.cam.resolutionLimit()
        hStep = resLimit.limitH.step
        hMin = resLimit.limitH.min
        hMax = resMax.height
        wStep = resLimit.limitW.step
        wMin = resLimit.limitW.min
        wMax = resMax.width

        resW = range(wMin, wMax+1, wStep if wStep != 0 else 1)
        resH = range(hMin, hMax+1, hStep if hStep != 0 else 1)
        resValues = []
        for h in resH:
            for w in resW:
                resValues.append(str(w)+"x"+str(h))
        self.resolutionList.config(values=resValues)
        res = self.cam.resolution()
        self.resolutionList.set(str(res.width)+"x"+str(res.height))

    def updateShutterList(self):
        fps = self.cam.framerate()
        shutValues = []
        for s in self.framerateValues:
               if s >= fps:
                   shutValues.append("1/" + str(s))
        self.shutterList.config(values = shutValues)
        shutter = self.cam.shutter()
        self.shutterList.set("1/" + str(shutter))

    def updateAcquisition(self):
        acq = self.acqutionVal.get()
        if acq and not self.cam.isXferring():
            self.cam.beginXfer(self.cppCallback)
        if not acq and self.cam.isXferring():
            self.cam.endXfer()

    def cppCallback(self, xferData):
        self.locker.acquire()
        if self.isRec:
            self.fcreator.write(xferData)
        self.locker.release()

    def rec(self):
        self.locker.acquire()
        self.isRec = not self.isRec
        if self.isRec:
            self.recButton.state(["pressed"])
            self.fcreator = FileCreator(str(BASE_DIR / "test"), self.savefileVal.get())
        else:
            self.recButton.state(["!pressed"])
            self.fcreator.close()
            FileCreator.create_json(str(BASE_DIR / "test"), self.cam)
        self.locker.release()

    def uistop(self):
        if self.uistopVal.get():
            self.after_cancel(self.updateID)
        else:
            self.updateID = self.after(self.delay, self.update)

    def terminate(self):
        self.after_cancel(self.updateID)
        self.cam.close()
        if self.fcreator is not None:
            self.fcreator.close()

    def resetSequenceNo(self):
        self.cam.resetSequenceNo()

    def resetDevice(self):
        self.locker.acquire()
        self.cam.resetDevice()
        self.cam = CameraFactory().create()
        self.framerateList.set(self.cam.framerate())
        self.updateResolutionList()
        self.updateShutterList()
        self.locker.release()


        #グラフの情報
        #グラフの初期化
    def create_graph(self):
        self.fig,self.ax = plt.subplots(figsize=(4,3),dpi=100)

        #グラフの初期化
        #x軸の幅を設定、時間軸、1フレームごと
        graph_x = np.arange(self.first_frame_num, self.last_frame_num)
        graph_y = np.zeros(self.draw_frame_num)

        
        #最初の表示部分
        self.lines_r_ankle, = self.ax.plot(graph_x, graph_y, color="#ff6347", label="Angle R Ankle")
        self.lines_r_knee, = self.ax.plot(graph_x, graph_y, color="#ffa500", label="Angle R Knee")
        self.lines_l_ankle, = self.ax.plot(graph_x, graph_y, color="#40e0d0", label="Angle L Ankle")
        self.lines_l_knee, = self.ax.plot(graph_x, graph_y, color="#90ee90", label="Angle L Knee")

        self.ax.set_title("joint degree")
        self.ax.set_xlabel("frame number")
        self.ax.set_ylabel("angle[degrees]")

        self.ax.set_xlim(self.first_frame_num, self.last_frame_num + 1)
        self.ax.set_ylim(0, 180)

        self.ax.legend(loc="lower left")


        self.canvas_g = FigureCanvasTkAgg(self.fig,master=self.graph)
        self.canvas_g.draw()
        self.canvas_g.get_tk_widget().pack(fill=tk.BOTH,expand=True)


    #グラフの更新
    def update_graph(self,pose_result, contact):

        if contact:
            self.contact_xlist.append(pose_result["last_frame_num"] - 1)
        # print("detect2")

        graph_x = np.arange(pose_result["first_frame_num"], pose_result["last_frame_num"])

        self.lines_r_ankle.set_data(graph_x, pose_result["right_ankle_angles_traj"])
        self.lines_r_knee.set_data(graph_x, pose_result["right_knee_angles_traj"])
        self.lines_l_ankle.set_data(graph_x, pose_result["left_ankle_angles_traj"])
        self.lines_l_knee.set_data(graph_x, pose_result["left_knee_angles_traj"])

        self.ax.vlines(self.contact_xlist, 0, 180, color="palevioletred")
        self.ax.set_xlim(graph_x.min(), graph_x.max())

        self.canvas_g.draw_idle()


    def update_angle_text_and_draw(self, pose_result, small_frame, toe_positions):
        self.canvas_window.create_text(self.pos[0]+5, self.pos[1]+5, anchor="nw", 
                                        text=f"Angle R Ankle: {int(pose_result["right_ankle_angles_list"][-1])}",
                                        font=("Arial", 15), fill="white")
        self.canvas_window.create_text(self.pos[0]+5, self.pos[1]+25, anchor="nw", 
                                        text=f"Angle R Knee: {int(pose_result["right_knee_angles_list"][-1])}",
                                        font=("Arial", 15), fill="white")
        self.canvas_window.create_text(self.pos[0]+5, self.pos[1]+45, anchor="nw", 
                                        text=f"Angle L Ankle: {int(pose_result["left_ankle_angles_list"][-1])}",
                                        font=("Arial", 15), fill="white")
        self.canvas_window.create_text(self.pos[0]+5, self.pos[1]+65, anchor="nw", 
                                        text=f"Angle L Knee: {int(pose_result["left_knee_angles_list"][-1])}",
                                        font=("Arial", 15), fill="white")

        r = 4
        self.canvas_window.create_oval(int(pose_result["joint_pixcels"][PoseAnalyzer.RIGHT_ANKLE][0]) + self.pos[0] - r , int(pose_result["joint_pixcels"][PoseAnalyzer.RIGHT_ANKLE][1]) + self.pos[1] - r, int(pose_result["joint_pixcels"][PoseAnalyzer.RIGHT_ANKLE][0]) + self.pos[0] + r , int(pose_result["joint_pixcels"][PoseAnalyzer.RIGHT_ANKLE][1]) + self.pos[1] + r,  fill="#ff6347")
        self.canvas_window.create_oval(int(pose_result["joint_pixcels"][PoseAnalyzer.RIGHT_KNEE][0]) + self.pos[0] - r , int(pose_result["joint_pixcels"][PoseAnalyzer.RIGHT_KNEE][1]) + self.pos[1] - r, int(pose_result["joint_pixcels"][PoseAnalyzer.RIGHT_KNEE][0]) + self.pos[0] + r , int(pose_result["joint_pixcels"][PoseAnalyzer.RIGHT_KNEE][1]) + self.pos[1] + r,  fill="#ffa500")
        self.canvas_window.create_oval(int(pose_result["joint_pixcels"][PoseAnalyzer.LEFT_ANKLE][0]) + self.pos[0] - r , int(pose_result["joint_pixcels"][PoseAnalyzer.LEFT_ANKLE][1]) + self.pos[1] - r, int(pose_result["joint_pixcels"][PoseAnalyzer.LEFT_ANKLE][0]) + self.pos[0] + r , int(pose_result["joint_pixcels"][PoseAnalyzer.LEFT_ANKLE][1]) + self.pos[1] + r,  fill="#40e0d0")
        self.canvas_window.create_oval(int(pose_result["joint_pixcels"][PoseAnalyzer.LEFT_KNEE][0]) + self.pos[0] - r , int(pose_result["joint_pixcels"][PoseAnalyzer.LEFT_KNEE][1]) + self.pos[1] - r, int(pose_result["joint_pixcels"][PoseAnalyzer.LEFT_KNEE][0]) + self.pos[0] + r , int(pose_result["joint_pixcels"][PoseAnalyzer.LEFT_KNEE][1]) + self.pos[1] + r,  fill="#90ee90")

        for toe_position in toe_positions:
            self.canvas_window.create_oval(int(toe_position[0]) + self.pos[0] - r , int(toe_position[1]) + self.pos[0] - r , int(toe_position[0]) + self.pos[0] + r , int(toe_position[1]) + self.pos[0] + r ,  fill="red")
                



    def update_contact_count_label(self, contact):

        self.posture_judge_label.config(
            text=f"Lifting Count: {self.contact_counter.contact_count}"
        )

    def update_height_judge_label(self, is_too_high):
        if is_too_high:
            self.height_judge_label.config(
                text=f"BALL IS TOO HIGH"
            )
        else:
            self.height_judge_label.config(
                text=f""
                )
      

        

        









    def update_judge_height(self,code):
        if code == True:
            self.judge_height("失敗")
        if code == False:
            self.judge_height("成功")

    def update_judge_position(self,code):
        if code == True:
            self.judge_position("成功")
        if code == False:
            self.judge_position("失敗")




    
    


class FileApplication(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)
        master.geometry("800x600")
        self.pack(expand=1, fill=tk.BOTH, anchor=tk.NW)

        self.font = tkfont.Font(self,family="Arial",size=10,weight="bold")

        self.createWidget()

    def createWidget(self):
        #---------------------------------------------------
        # options frame
        #---------------------------------------------------
        frameWidth=300
        self.optionFrame = ttk.LabelFrame(self, 
                                          text="file data", 
                                          width=frameWidth,
                                          relief=tk.RAISED)
        self.optionFrame.propagate(False)
        self.optionFrame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        #---------------------------------------------------
        # file open
        #---------------------------------------------------
        self.framecountPanel = ttk.Frame(self.optionFrame,
                                        width=frameWidth,
                                        height=30,
                                        relief=tk.FLAT)

        self.framecountPanel.propagate(False)
        self.framecountPanel.pack(side=tk.TOP, fill=tk.Y, padx=5, pady=5)

        self.framecount_text = tk.StringVar()
        self.framecount_text.set("file framecount = %d" % 0)
        self.fileframelabel = tk.Label(self.framecountPanel, textvariable = self.framecount_text)
        self.fileframelabel.pack(side=tk.LEFT, padx=5)

        self.filePanel = ttk.Frame(self.optionFrame,
                                  width=frameWidth,
                                  height=30,
                                  relief=tk.FLAT)

        self.filePanel.propagate(False)
        self.filePanel.pack(side=tk.BOTTOM, fill=tk.Y, padx=5, pady=5)


        self.fileButton = ttk.Button(self.filePanel, 
                                    text = "OPEN", 
                                    command=self.openfile,
                                    width=15)
        self.fileButton.pack(side=tk.RIGHT,anchor="center", expand=True)



        #---------------------------------------------------
        # canvas
        #---------------------------------------------------
        self.canvas = tk.Canvas(self, width=1296, height=1080)
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)



    def createimagedata(self, data, seqNo):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        w = self.reader.width
        h = self.reader.height
        scale = 1
        if cw > 1 and ch > 1:
            scale = cw/w if cw/w < ch/h else ch/h

        array = data
        i = Image.fromarray(array).resize((int(w*scale), int(h*scale)))
        self.img = ImageTk.PhotoImage(image=i)
        self.canvas.delete("all")
        pos = [(cw-i.width)/2,(ch-i.height)/2]
        self.canvas.create_image(pos[0], pos[1], anchor="nw", image=self.img)
        self.canvas.create_text(pos[0]+5, pos[1]+5, anchor="nw", 
                                text="SequeceNo:" + str(seqNo + self.iniFileSeqNo),
                                font=self.font, fill="limeGreen")
        
    def openfile(self):
        dir = os.path.abspath(os.path.dirname(__file__))
        type = [("データファイル","*.json")]
        fname = filedialog.askopenfilename(filetypes=type, initialdir=dir)

        self.reader = BinaryReader(fname)
        data = self.reader.read(0)
        self.iniFileSeqNo = self.reader.readseqNo(0)
        
        self.createimagedata(data, self.iniFileSeqNo)

        self.filespinboxLabel = ttk.Label(self.filePanel, text="Frame:", width=8)
        self.filespinboxLabel.pack(side=tk.LEFT, padx=5)

        self.filespinBox = ttk.Spinbox(self.filePanel,
                                       textvariable=0,
                                       from_=0,
                                       to=self.reader.framecount,
                                       command=self.updatecanvas,
                                       increment=1,
                                       )
        self.filespinBox.pack(side=tk.LEFT, padx=5)
        
        self.framecount_text.set("file framecount = %d" % self.reader.framecount)


    def updatecanvas(self):
        seqNo = int(self.filespinBox.get())
        data = self.reader.read(seqNo)
        self.createimagedata(data, seqNo)





def main():
    global root,notebook,tframes,fnames
    root = tk.Tk()

    root.title('tabbed editor')
    root.geometry('800x600')
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both',expand=1)



    camapp = CamApplication(master = root)
    notebook.add(camapp, text="cam")
    fileapp = FileApplication(master = root)
    notebook.add(fileapp, text="file")



   

    camapp.mainloop()
    camapp.terminate()

if __name__ == '__main__':
    main()