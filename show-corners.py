import numpy as np
import cv2
import glob

# How many rows and columns does the images of your checkerboard show?
ROWS = 10
COLS = 13

# The code is actually looking for the _available corners_, so we can simply
# subtract one from the rows and columns value.
HORZ_CORNERS = COLS-1
VERT_CORNERS = ROWS-1

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((HORZ_CORNERS*VERT_CORNERS,3), np.float32)
objp[:,:2] = np.mgrid[0:VERT_CORNERS,0:HORZ_CORNERS].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('samples/checkerboard-*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (VERT_CORNERS,HORZ_CORNERS),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (VERT_CORNERS,HORZ_CORNERS), corners2,ret)
        cv2.imshow('img', img)
        cv2.waitKey(5000)

cv2.destroyAllWindows()
