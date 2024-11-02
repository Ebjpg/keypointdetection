import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame, save=True, conf=0.1)

        annotated_frame = results[0].plot()

        cv2.imshow("Yolov8", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
