import cv2
from ultralytics import YOLO


model = YOLO('yolov8n-pose.pt')


video_path = 'enter your video location'
output_path = 'enter output video name'


cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video format

# VideoWriter nesnesini oluştur
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame, conf=0.1)

        annotated_frame = results[0].plot()

        out.write(annotated_frame)

        cv2.imshow("Yolov8", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
