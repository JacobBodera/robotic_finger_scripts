import os
import cv2 as cv
import time

images = []

for image in os.listdir('test_images\\'):
    images.append(cv.imread(f'test_images\\{image}'))

time.sleep(5)
cv.imwrite('outputtest.png', images[0])
