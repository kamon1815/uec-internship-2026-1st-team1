import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
video_path = BASE_DIR / "../movie/zikken3.avi"

class VideoPlayer:

    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("Video Player")
        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window)
        self.canvas.grid(row=0, column=0, columnspan=5)

        self.btn_rewind = tk.Button(window, text="<< 5s", width=10, command=self.rewind)
        self.btn_rewind.grid(row=1, column=0, sticky="ew")

        self.btn_play_pause = tk.Button(window, text="Play", width=10, command=self.toggle_play)
        self.btn_play_pause.grid(row=1, column=1, sticky="ew")

        self.btn_skip = tk.Button(window, text="5s >>", width=10, command=self.skip)
        self.btn_skip.grid(row=1, column=2, sticky="ew")

        self.progress_bar = ttk.Progressbar(window, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.grid(row=1, column=3, sticky="ew")

        self.lbl_timestamp = tk.Label(window, text="00:00/00:00")
        self.lbl_timestamp.grid(row=1, column=4, sticky="ew")

        self.paused = True
        self.update()

    def toggle_play(self):
        self.paused = not self.paused
        if self.paused:
            self.btn_play_pause.config(text="Play")
        else:
            self.btn_play_pause.config(text="Pause")
            self.update()

    def rewind(self):
        current_frame = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))
        fps = self.vid.get(cv2.CAP_PROP_FPS)
        target_frame = max(current_frame - 5 * fps, 0)
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    def skip(self):
        current_frame = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.vid.get(cv2.CAP_PROP_FPS)
        target_frame = min(current_frame + 5 * fps, total_frames)
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    def update(self):
        if not self.paused:
            ret, frame = self.vid.read()
            if ret:
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.config(width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                current_frame = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))
                total_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
                self.progress_bar["value"] = (current_frame / total_frames) * 100
                current_time = int(self.vid.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                total_time = int(total_frames / self.vid.get(cv2.CAP_PROP_FPS))
                current_time_str = self.format_time(current_time)
                total_time_str = self.format_time(total_time)
                self.lbl_timestamp.config(text=f"{current_time_str}/{total_time_str}")
                fps = self.vid.get(cv2.CAP_PROP_FPS)
                delay = int(1000 / fps)
                self.window.after(delay, self.update)
            else:
                self.toggle_play()
        else:
            self.btn_play_pause.config(text="Play")

    def format_time(self, seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"


root = tk.Tk()
player = VideoPlayer(root, video_path)
root.mainloop()
