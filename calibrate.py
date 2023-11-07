'''
Name:          Jacob Bodera
Date:          October 25, 2023
Description     
'''

'''     IMPORTS     '''
import cv2 as cv
import numpy as np
import time
import pickle

'''     FUNCTIONS       '''
def countDown(t):
    for j in range(t):
        t = float(t)
        print(j+1)
        time.sleep(1)

def vidCap():
    cap = cv.VideoCapture(1, cv.CAP_DSHOW)
    ret, frame = cap.read()
    cv.imshow('im', frame)
    time.sleep(0.5)
    return frame

'''     CONSTANTS       '''
# number of images that are used for calibration
num_iterations = 5
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# size of checkboard - need to omit 1 from each dimension for border
grid_height = 4
grid_width = 7
square_size = 0.5 # cm

'''     PROGRAM     '''
cv.destroyAllWindows()

# matrix to hold coordinates of the checkerboard
objp = np.zeros((grid_height*grid_width,3), np.float32)
objp[:,:2] = np.mgrid[0:grid_width, 0:grid_height].T.reshape(-1,2)
objp = objp*square_size

# lists to store object points and image points from all the images
object_points = [] # 3d points in real space
image_points = [] # 2d points in image plane

for iteration in range(num_iterations):
    countDown(5)
    img = vidCap()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # find chess board corners
    grid_found, grid_corners = cv.findChessboardCorners(gray, (grid_width, grid_height), None)
    print('Grid found: ', grid_found)
    # if found, add object points, image points
    if grid_found == True:
        object_points.append(objp)
        grid_corners2 = cv.cornerSubPix(gray, grid_corners, (11,11), (-1,-1), criteria)
        image_points.append(grid_corners)
        cv.drawChessboardCorners(img, (grid_width,grid_height), grid_corners2, grid_found)
        cv.imshow('img', img)
        cv.waitKey(500)

# get intristic parameters of camera
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

# removing distortion
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imshow('dst', dst)

# check the calibration and compute the error 
mean_error = 0
for i in range(len(object_points)):
    image_points2, _ = cv.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(image_points[i], image_points2, cv.NORM_L2)/len(image_points2)
    mean_error += error
print( "total error: {}".format(mean_error/len(object_points)) )


# Save camera's intrinsic matrix to text file
# object_points and image_points are pickled and saved as numpy array in .npy files for use in other programs
np.savetxt('CamIntrMtx.txt', mtx)

with open("objpoints", "wb") as fp:   #Pickling 
    pickle.dump(object_points, fp)
    
with open("imgpoints", "wb") as gp:   #Pickling 
    pickle.dump(image_points, gp)