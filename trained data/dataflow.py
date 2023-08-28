from roboflow import Roboflow
import cv2

rf = Roboflow(api_key="GpNgni7iV7aCPFEJNQuK")
project = rf.workspace().project("play-together-traing")
model = project.version(1).model

# Đọc ảnh và lấy kích thước ảnh
image_path = "check.png"
video_path = "cauca.mp4"
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape

# Infer on a local image
data_detect = model.predict(image_path).json()[
    'predictions']
model.predict(video_path, confidence=40, overlap=30).save("video.mp4")

# Vòng lặp để xử lý và hiển thị các bounding box
for detect in data_detect:
    print(detect)
    x, y, width, height = detect['x'], detect['y'], detect['width'], detect['height']
    classter = detect['class']
    confidence = detect['confidence']

    # Vẽ bounding box
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(image, f"{classter} ({confidence:.2f})",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Hiển thị ảnh sau khi vòng lặp đã xử lý xong
cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
