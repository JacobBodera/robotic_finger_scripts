import cv2 as cv


camera = cv.VideoCapture(1)


value, image = camera.read()
cv.imwrite('test_images/test.png', image)
del(camera)
