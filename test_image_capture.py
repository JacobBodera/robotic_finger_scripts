import cv2 as cv
import time


camera = cv.VideoCapture(1)

time.sleep(7)
value, image = camera.read()
cv.imwrite('test_images/test2.png', image)

del(camera)
 