import numpy as np
import cv2

# How many rows and columns does the images of your checkerboard show?
ROWS = 10
COLS = 7

# The code is actually looking for the _available corners_, so we can simply
# subtract one from the rows and columns value.
HORZ_CORNERS = COLS-1
VERT_CORNERS = ROWS-1

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((HORZ_CORNERS*VERT_CORNERS,3), np.float32)
objp[:,:2] = np.mgrid[0:VERT_CORNERS,0:HORZ_CORNERS].T.reshape(-1,2)


def run(channel):
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # The number we pass is the camera number starting with 0 (typically for a
    # built-in camera)
    cap = cv2.VideoCapture(channel)

    mtx = None
    dist = None
    newcameramtx = None

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (VERT_CORNERS,HORZ_CORNERS),None)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        if key & 0xFF == ord('a') and ret == True:
            # If found, add object points, image points (after refining them)
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            gray = cv2.drawChessboardCorners(gray, (VERT_CORNERS,HORZ_CORNERS), corners2,ret)
            print("calculating calibration... %d" % (len(objpoints)))
            # mtx: camera matrix
            # dist: distortion coefficients
            # rvecs: rotation vectors
            # tvecs: translation vectors
            (w,h) = gray.shape[::-1]
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w,h),None,None)

            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        if mtx is not None:
            print("undistorting")
            gray = cv2.undistort(gray, mtx, dist, None, newcameramtx)

        cv2.imshow('frame',gray)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


run(1)
