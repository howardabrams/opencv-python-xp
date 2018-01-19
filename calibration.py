#!/usr/bin/env python
import numpy as np
import cv2
import argparse


def run(channel, rows, cols, filename):
    """
    Generate calibration data file based on viewing a checkerboard pattern with
a specific number of rows and columns.

    channel :: the USB port channel to use when accessing a camera
    rows    :: the number of rows in the _expected_ checkerboard pattern
    cols    :: the number of _expected_ columns in pattern
    filename:: name of the file to store the cached calibration values
    """
    # The code is actually looking for the _available corners_, so we can simply
    # subtract one from the rows and columns value.
    horz_corners = cols-1
    vert_corners = rows-1

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((horz_corners*vert_corners, 3), np.float32)
    objp[:, :2] = np.mgrid[0:vert_corners, 0:horz_corners].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

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
        ret, corners = cv2.findChessboardCorners(gray, (vert_corners, horz_corners), None)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        if key & 0xFF == ord('a') and ret:
            # If found, add object points, image points (after refining them)
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            gray = cv2.drawChessboardCorners(gray, (vert_corners, horz_corners), corners2, ret)
            print("calculating calibration... %d" % (len(objpoints)))
            # mtx: camera matrix (includes focal length and optical centers)
            # dist: distortion coefficients
            # rvecs: rotation vectors
            # tvecs: translation vectors
            (w, h) = gray.shape[::-1]
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        if mtx is not None:
            print("undistorting")
            gray = cv2.undistort(gray, mtx, dist, None, newcameramtx)

        if key & 0xFF == ord('s'):
            np.savez(filename, mtx=mtx, dist=dist, newcammtx=newcameramtx)
            break

        cv2.imshow('frame', gray)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create calibration values file by imaging a checkerboard pattern')
    parser.add_argument('-p', '--channel', default=1, type=int,
                        help='the USB channel containing camera, 0, 1, or 2')
    parser.add_argument('-r', '--rows', default=10, type=int,
                        help='the number of expected rows on our checkerboard pattern')
    parser.add_argument('-c', '--columns', default=7, type=int,
                        help='the number of columns expected on our checkerboard pattern')
    parser.add_argument('-o', '--output', default="calibration-values",
                        help='filename to contain the calibration values')

    args = parser.parse_args()

    print("""
  Position a black and white checkerboard pattern with {} rows and {} columns
  in front of your USB-enabled camera. Press the 'a' key with the pattern in
  each of the four corners of the displayed camera image. When the image is
  reasonably straight, press the 's' key to save the calibration values to a
  file that the other programs can read.

  Press 'q' to cancel and quit this application. *Note:* The keys must be
  pressed with the image window is the foremost window, otherwise, you can
  cancel this application with Control-C.
    """.format(args.rows, args.columns))

    run(args.channel, args.rows, args.columns, args.output)
