import cv2

class VideoProcessor:
    def __init__(self,video_path,output_path):
        self.cap = cv2.VideoCapture(video_path)

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (width,height)
        )

    def close(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()