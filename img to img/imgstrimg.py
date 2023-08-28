import cv2
import numpy as np
import matplotlib.pyplot as plt
pic1 = "target.png"
pic2 = "oop.png"
img_rgb = cv2.imread(pic1)
template = cv2.imread(pic2)
h = template.shape[0]
w = template.shape[1]
check = False
res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
threshold = .8
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):  # Switch collumns and rows
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

cv2.imshow('Worm Detection Mark', cv2.resize(img_rgb, (400, 200)))
cv2.waitKey(0)
# cv2.imwrite('result.png', img_rgb)

