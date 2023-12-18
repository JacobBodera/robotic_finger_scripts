'''
Name:           Jacob Bodera
Date:           October 2023
Description:    This file shows how to use opencv to extract images from a directory and append them to a python list
'''

import os
import cv2 as cv
import time

images = []

for image in os.listdir('test_images\\'):
    images.append(cv.imread(f'test_images\\{image}'))

time.sleep(5)
cv.imwrite('outputtest.png', images[0])
