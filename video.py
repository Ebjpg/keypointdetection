import cv2
from ultralytics import YOLO

# YOLOv8n-pose modelini yükle
model = YOLO('yolov8n-pose.pt')

# Video dosyasının yolunu belirtin
video_path = 'Cinematic Tennis Commercial - Shot on Sony FX3.mp4'  # Buraya video dosyanızın tam yolunu girin
output_path = 'output_video.mp4'  # Tahmin yapılmış video dosyasının kaydedileceği yol

# VideoCapture nesnesini video dosyasından oluşturun
cap = cv2.VideoCapture(video_path)

# Video özelliklerini al
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 formatı için

# VideoWriter nesnesini oluştur
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Modeli çerçeve üzerinde çalıştır ve minimum güven seviyesini belirleyin
        results = model(frame, conf=0.1)

        # Tahminleri çerçeveye çiz
        annotated_frame = results[0].plot()

        # Annotated çerçeveyi video dosyasına yaz
        out.write(annotated_frame)

        # Görüntüyü ekranda göster
        cv2.imshow("Yolov8", annotated_frame)

        # 'q' tuşuna basıldığında döngüden çık
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

# VideoCapture ve VideoWriter'ı serbest bırak
cap.release()
out.release()
cv2.destroyAllWindows()
