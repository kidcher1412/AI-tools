import cv2

cap = cv2.VideoCapture('http://127.0.0.1:5000/video')

while True:
    res,frame = cap.read()
    if res == 0:
        continue
    else:
        cv2.imshow("Test video stream",frame)
        k = cv2.waitKey(20) & 0xFF
        if k == 27 :
            break
cap.release()
cv2.destroyAllWindows()