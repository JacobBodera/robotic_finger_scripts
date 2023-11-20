'''
Name:           Jacob Bodera, Alec
Date:           November 2nd, 2023
Description     
'''

'''     IMPORTS     '''
import numpy as np
import cv2 as cv
import math
import time
# import matplotlib
import pickle
import pyfirmata
import serial.tools.list_ports
import imutils
from timeit import default_timer as timer
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist
from PIL import Image
import os

'''     FUNCTIONS       '''

def photoShoot(IO, t):
    v = []
    photos = []
    cam = cv.VideoCapture(0, cv.CAP_DSHOW)

    i = 0
    while True:
        i = i + 1
        print(i)
        ret, frame = cam.read()
        photos.append(frame)
        time.sleep(t)
        
        if i ==10:
            IO = False
        
        if IO == False:
            break        
        
    return photos, v

def resize(file, scale):
    RowIm = int(file.shape[0] * scale / 100)
    ColIm = int(file.shape[1] * scale / 100)
    dim = (ColIm, RowIm)
    resized = cv.resize(file, dim)
    return resized

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    
def ShapeDet(images, ref_Width):
    lineLen = []
    strainImages = []
    XList = []
    YList = []

    for i in range(len(images)):
        print(i)
        image = images[i]
        lineLen.append([])

        # Convert from color to HUE, SATURATION, VALUE
        img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        # Lower color mask
        lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])
        mask0 = cv.inRange(img_hsv, lower_red, upper_red)

        # Upper color mask
        lower_red = np.array([170,50,50])
        upper_red = np.array([180,255,255])
        mask1 = cv.inRange(img_hsv, lower_red, upper_red)

        # Combine the two masks to get clearer mask of red
        mask = mask0 + mask1

        # Set output image pixels to zero except mask
        output_img = image.copy()
        output_img[np.where(mask==0)] = 0

        # Do the same for the HSV image
        output_hsv = img_hsv.copy()
        output_hsv[np.where(mask==0)] = 0

        rgb = cv.cvtColor(output_img, cv.COLOR_HSV2RGB)	
        gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
        
        (_, gray) = cv.threshold(gray, 115, 255, cv.THRESH_BINARY)

        gray - cv.GaussianBlur(gray, (7, 7), 0)
        
        # Perform edge detection and close gaps
        edged = cv.Canny(gray, 50, 100)
        edged = cv.dilate(edged, None, iterations=1)
        edged = cv.erode(edged, None, iterations=1)
        # find contours in the edge map
        cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # Sort contours from left->right and initialize calibration variable
        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None

        xMidPoints = []
        yMidpoints = []
        original = image.copy()
        # cv.imshow("Image", original)

        k = 0
        for cnt in cnts:
            # ignore if contour is not significantly large enough
            if cv. contourArea(cnt) < 50:
                continue

            xMidPoints.append([])
            yMidpoints.append([])

            # compute the rotating bounding box of contour
            box = cv.minAreaRect(cnt)
            box = cv.boxPoints(box)
            box = np.array(box, dtype="int")

            # order the points in contour in top-left, top-right, bottom-right, bottom-left order
            # draws rotated bounding box
            box = perspective.order_points(box)
            cv.drawContours(original, [box.astype("int")], -1, (0, 255, 0), 2)
            # loop over original points and draw them
            for (x, y) in box:
                cv.circle(original, (int(x), int(y)), 5, (0, 0, 255), -1)

            # unpack ordererd bounding box, compute midpoint between topLeft & topRight, and bottomLeft & bottomRight
            tLeft, tRight, bRight, bLeft = box
            tltrX, tltrY = midpoint(tLeft, tRight)
            blbrX, blbrY = midpoint(bLeft, bRight)
            # midpoint between topLeft & bottomLeft, and topRight & bottomRight
            tlblX, tlblY = midpoint(tLeft, bLeft)
            trbrX, trbrY = midpoint(tRight, bRight)

            # define midpoint of contour
            contour_midX = (tltrX + blbrX) / 2
            contour_midY = (tlblY + trbrY) / 2

            xMidPoints[k].append(contour_midX)
            yMidpoints[k].append(contour_midY)

            # draw midpoints on the image
            cv.circle(original, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv.circle(original, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv.circle(original, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv.circle(original, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
            cv.circle(original, (int(contour_midY), int(contour_midY)), 5, (255, 0, 255), -1)
            # draw lines between the midpoints
            cv.line(original, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
            cv.line(original, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

            if len(xMidPoints) > 1:
                cv.circle(original, (int(xMidPoints[k][len(xMidPoints[k])-2]), int(yMidpoints[k][len(yMidpoints[k])-2])), 5, (0, 255, 0), -1)
                cv.line(original, (int(xMidPoints[len(xMidPoints)-2][0]), int(yMidpoints[len(yMidpoints)-2][0])), (int(xMidPoints[len(xMidPoints)-1][0]), int(yMidpoints[len(yMidpoints)-1][0])), (0, 0, 0), 2)
                lineLen[i].append(dist.euclidean((xMidPoints[k][len(xMidPoints[k])-2], yMidpoints[k][len(yMidpoints[k])-2]),(xMidPoints[k][len(xMidPoints[k])-1], yMidpoints[k][len(yMidpoints[k])-1]) ))   

                strainImages.append(original)

                if i == 0:
                    start_len = dist.euclidean((xMidPoints[k][len(xMidPoints[k])-2], yMidpoints[k][len(yMidpoints[k])-2]),(xMidPoints[k][len(xMidPoints[k])-1], yMidpoints[k][len(yMidpoints[k])-1]))
                    # if pixels per metrix is not initialized, compute ration of pixels to supplied metrix (inches in this case)
                    perMetric = start_len / ref_Width
                
                if pixelsPerMetric != None:
                    adj_len = lineLen[-1][-1] / pixelsPerMetric

                    cv.putText(original, "{:.1f}in".format(adj_len), ((xMidPoints[c][len(xMidPoints[c])-2] + xMidPoints[k][len(xMidPoints[k])-1])/2, (yMidpoints[k][len(yMidpoints[k])-2] + yMidpoints[k][len(yMidpoints[k])-1])/2), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            # compute the Euclidian distance between the midpoints
            distA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            distB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            cv.waitKey(3000)
            cv.destroyAllWindows()
            k = k + 1
        
        # cv.imshow("image", original)
        XList.append(xMidPoints)
        YList.append(yMidpoints)

    return XList, YList, strainImages


'''     SETUP       '''

# Loading camera matrix from 'calibration.py'
# This matrix is the intrinsic distortion of the camera
mtx = np.loadtxt("CamIntrMtx.txt", dtype=str)

# Unpickling
with open("objpoints", "rb") as fp:
    # 3D points in real space
    objpoints = pickle.load(fp)
with open("imgpoints", "rb") as gp:
    # 2D points on image plane
    imgpoints = pickle.load(gp)  

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


grid_height = 4
grid_width = 7 
square_size  = 5.0

#----matrix to hold the coordinates of the checkerboard
objp = np.zeros((grid_height*grid_width,3), np.float32)
objp[:,:2] = np.mgrid[0:grid_width, 0:grid_height].T.reshape(-1,2)
objp = objp*square_size

'''     PROGRAM     '''

time.sleep(5)

images = []

for image in os.listdir('test_images\\'):
    images.append(cv.imread(f'test_images\\{image}'))

refWidth = 10

xList, yList, def1Images = ShapeDet(images, refWidth)

# xArr = np.array(xList)
# yArr = np.array(yList)
# square = math.ceil(math.sqrt(len(images)))

print('get here')
print(def1Images)

imNum = 0
for image in def1Images:
    cv.imwrite(f'output_images/out_image{imNum}.png', image)
    imNum += 1

# figure, axis = matplotlib.pyplot.subplot(square, square)
# P = []

# for i in range(square):
#     for j in range(square):
#         if (k < len(images)):
#             axis[i, j].plot(xArr[k], -yArr[k], '.')
#         k = k + 1




