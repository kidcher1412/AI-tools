import cv2
import numpy as np

# Đường dẫn đến tệp deploy.prototxt và weights.caffemodel
prototxt_path = 'lib/deploy.prototxt'
caffemodel_path = 'lib/weights.caffemodel'

# Đọc mô hình từ tệp deploy.prototxt và weights.caffemodel
net = cv2.dnn.readNet(prototxt_path, caffemodel_path)

# Đọc ảnh đầu vào
image_path = 'target.jpg'
image = cv2.imread(image_path)

# Chuẩn bị ảnh để đưa vào mô hình
blob = cv2.dnn.blobFromImage(image, scalefactor=0.007843, size=(
    300, 300), mean=(127.5, 127.5, 127.5), swapRB=True, crop=False)

# Đưa ảnh qua mạng để phát hiện đối tượng
net.setInput(blob)
detections = net.forward()

# Vẽ bounding box cho các đối tượng được phát hiện
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:  # Ngưỡng độ tin cậy
        class_id = int(detections[0, 0, i, 1])
        left = int(detections[0, 0, i, 3] * image.shape[1])
        top = int(detections[0, 0, i, 4] * image.shape[0])
        right = int(detections[0, 0, i, 5] * image.shape[1])
        bottom = int(detections[0, 0, i, 6] * image.shape[0])

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f'Class {class_id}: {confidence:.2f}'
        cv2.putText(image, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Hiển thị hình ảnh với bounding box
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
