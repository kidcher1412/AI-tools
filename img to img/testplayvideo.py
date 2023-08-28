import cv2
import numpy as np

# Đường dẫn đến tệp ảnh chứa con sâu
video_path = 'cauca.mp4'

# # Đọc ảnh từ đường dẫn
# cap = cv2.VideoCapture(video_path)
# while cap.isOpened():
#     # Đọc khung hình từ video
#     ret, frame = cap.read()
#     if not ret:
#             break
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


#     # Đặt ngưỡng cho màu sâu
#     lower_green = np.array([94, 111, 189])
#     upper_green = np.array([179, 177, 220])

#     # Tạo ngưỡng binary
#     mask = cv2.inRange(hsv, lower_green, upper_green)

#     # Xử lý nhiễu
#     mask = cv2.medianBlur(mask, 5)

#     # Tìm contours
#     contours, _ = cv2.findContours(
#         mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Vẽ bounding box và xác định vùng con sâu
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area > 80:  # Chỉ xem xét các vùng có diện tích đủ lớn
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(frame, (x, y), (x + w, y + h),
#                         (0, 255, 0), 1)  # Giảm size nét vẽ
#             cv2.putText(frame, f'cá: {w*h}', (x, y - 15),  # Thêm kích thước vào văn bản
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)  # Giảm size nét chữ
#     #         M = cv2.moments(contour)
#     #         if M["m00"] != 0:
#     #             cX = int(M["m10"] / M["m00"])
#     #             cY = int(M["m01"] / M["m00"])
#     #             cv2.drawContours(image1, contours, -1, (0, 255, 0), 2)

#     # Hiển thị hình ảnh
#     cv2.imshow('Worm Detection', frame)

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    # Đọc khung hình từ video
    ret, image = cap.read()
    if not ret:
        break


    # Chuyển đổi ảnh sang không gian màu HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Đặt ngưỡng cho màu sâu
    # lower_green = np.array([94, 125, 189])
    # upper_green = np.array([100, 179, 210])
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
        if area > 1500:  # Chỉ xem xét các vùng có diện tích đủ lớn
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h),
                        (0, 0, 255), 1)  # Giảm size nét vẽ
            cv2.putText(image, f'cá: {area}', (x, y - 15),  # Thêm kích thước vào văn bản
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Giảm size nét chữ

    cv2.imshow("frame", image)

    # Chờ 1 milisecond và kiểm tra xem có phím nào được nhấn không
    key = cv2.waitKey(1)
    if key == 27:  # Khi nhấn phím Esc (mã ASCII 27)
        print('lower case:')
        print('uper case:')
        break  # Thoát khỏi vòng lặp