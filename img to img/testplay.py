import cv2
import numpy as np

# Đường dẫn đến tệp ảnh chứa con sâu
image_path = 'fish.jpg'

# Đọc ảnh từ đường dẫn
image = cv2.imread(image_path)
image1 = cv2.imread(image_path)

# Chuyển đổi ảnh sang không gian màu HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Đặt ngưỡng cho màu sâu
lower_green = np.array([85, 160, 191])
upper_green = np.array([133, 194, 205])

# Tạo ngưỡng binary
mask = cv2.inRange(hsv, lower_green, upper_green)

# Xử lý nhiễu
mask = cv2.medianBlur(mask, 5)

# Tìm contours
contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Vẽ bounding box và xác định vùng con sâu
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 80:  # Chỉ xem xét các vùng có diện tích đủ lớn
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h),
                      (0, 255, 0), 1)  # Giảm size nét vẽ
        cv2.putText(image, f'cá: {w*h}', (x, y - 15),  # Thêm kích thước vào văn bản
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)  # Giảm size nét chữ
#         M = cv2.moments(contour)
#         if M["m00"] != 0:
#             cX = int(M["m10"] / M["m00"])
#             cY = int(M["m01"] / M["m00"])
#             cv2.drawContours(image1, contours, -1, (0, 255, 0), 2)

# Hiển thị hình ảnh
cv2.imshow('Worm Detection', cv2.resize(image, (400, 200)))
cv2.imshow('Worm Detection Mark1', cv2.resize(image1, (400, 200)))
cv2.imshow('Worm Detection Mark2', cv2.resize(hsv, (400, 200)))
cv2.imshow('Worm Detection Mark3', cv2.resize(mask, (400, 200)))
cv2.waitKey(0)
cv2.destroyAllWindows()
