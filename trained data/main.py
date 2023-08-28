from roboflow import Roboflow
import cv2
import torch

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='playtogether.pt')

class_names = ['fish']

# Open camera
# cap = cv2.VideoCapture(0)  # Sử dụng camera thứ 0 (camera mặc định)
cap = cv2.VideoCapture('cauca.mp4')  # Sử dụng camera thứ 0 (camera mặc định)

display_interval = 3  # Hiển thị ảnh sau mỗi 3 lần quét mới
display_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if display_counter == 0:
        # Detect objects
        results = model(frame)

        # Get bounding box information
        bboxes = results.xyxy[0].cpu().numpy()

        # Hiển thị kết quả
        for bbox in bboxes:
            x1, y1, x2, y2, conf, cls = bbox
            class_name = class_names[int(cls)]
            label = f"Class: {class_name}, Confidence: {conf:.2f}"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(
                x2), int(y2)), (0, 255, 0), 2)  # Vẽ bounding box
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Vẽ chữ

        cv2.imshow('Object Detection', frame)
        display_counter = display_interval
    else:
        display_counter -= 1

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
