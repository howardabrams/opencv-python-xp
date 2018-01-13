import numpy as np

def calibration():
    """Returns the camera matrix, distortion, and optimal camera
    matrix values from the callibration file created by calibration.py"""

    cal = np.load("calibration-values.npz")
    return cal["mtx"], cal["dist"], cal["newcammtx"]

# Example: assuming we have an image saved
mtx, dist, newcammtx = calibration()
newimage = cv2.undistort(img, mtx, dist, None, newcammtx)
