import cv2 as cv


camera = cv.VideoCapture(1)

for i in range(5):
    value, image = camera.read()
    cv.imwrite(f'test_images/test{i}.png', image)

del(camera)
 