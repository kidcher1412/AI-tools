import cv2
import numpy as np

# Đường dẫn đến tệp ảnh chứa con sâu
image_path = 'con_sau1.jpg'

# Đọc ảnh từ đường dẫn
image = cv2.imread(image_path)
marked_image = image.copy()

# Chuyển đổi ảnh sang không gian màu HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Đặt ngưỡng cho màu sâu
lower_green = np.array([25, 50, 50])
upper_green = np.array([85, 255, 255])

# Tạo ngưỡng binary
mask = cv2.inRange(hsv, lower_green, upper_green)

# Xử lý nhiễu
mask = cv2.medianBlur(mask, 5)

# Tìm contours
contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Tạo lớp mặt nạ với độ trong suốt 30%
alpha = 0.3

# Vẽ màu đỏ nhạt với độ trong suốt 30% cho từng con sâu
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:  # Chỉ xem xét các vùng có diện tích đủ lớn
        mask_single_contour = np.zeros_like(mask)
        cv2.drawContours(mask_single_contour, [
                         contour], -1, 255, thickness=cv2.FILLED)

        # Tạo lớp mặt nạ cho vùng contour
        overlay = np.zeros_like(image)
        overlay[mask_single_contour == 255] = [0, 0, 255]

        # Tạo ảnh đỏ nhạt với độ trong suốt 30%
        overlay_blended = cv2.addWeighted(
            marked_image, 1 - alpha, overlay, alpha, 0)

        # Thay thế vùng contour bằng ảnh đỏ nhạt
        marked_image[mask_single_contour ==
                     255] = overlay_blended[mask_single_contour == 255]

# Hiển thị hình ảnh với vùng con sâu đã được xử lý
cv2.imshow('Processed Image', marked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
