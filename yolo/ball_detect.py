import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt") 
video_path = "C:\\Users\\intern02\\Desktop\\intern\\sample\\img\\ball.mp4"
cap = cv2.VideoCapture(video_path)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
#results = model("C:\\Users\\intern02\\Desktop\\intern\\sample\\img\\ball.mp4")

#for r in results:
    #boxes = r.boxes
    #for box in boxes:
        #cls_id = int(box.cls[0])
        #conf = float(box.conf[0])
        #if model.names[cls_id] == "sports ball":
            #x1,y1,x2,y2 = box.xyxy[0].tolist()
            #cx = (x1+x2)/2
            #cy = (y1+y2)/2
        #print(f"Detected:{model.names[cls_id]} with confidence{conf:.2f}")
            #print(f"中心座標:({cx},{cy})")

while cap.isOpened():
    success,frame = cap.read()
    if success:
        results = model(frame)

        imageWidth = results[0].orig_shape[0]
        imageHeight = results[0].orig_shape[1]

        names = results[0].names
        classes = results[0].boxes.cls
        boxes = results[0].boxes
        boxedframe = results[0].plot()

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if model.names[cls_id] == "sports ball":
                        x1,y1,x2,y2 = box.xyxy[0].tolist()
                        cx = (x1+x2)/2
                        cy = (y1+y2)/2
                        print(f"中心座標:({cx},{cy})")
                        cv2.circle(boxedframe, (int(cx),int(cy)), 10,(0,0,255), -1)

            cv2.imshow("center",boxedframe) 

            if cv2.waitKey(1) & 0xFF == ord("q"):
                  break
    else:
          break    

cap.release()
cv2.destroyWindow()       

frame.save(filename="result.avi")