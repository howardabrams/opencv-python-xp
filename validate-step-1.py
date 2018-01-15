#!/usr/bin/env python

import cv2

# Can we import the OpenCV library, cv2?
# If so, we are almost there ...

img = cv2.imread('samples/random-pix-1.jpg', 0)
cv2.imshow('image', img)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'):  # wait for 's' key to save and exit
    cv2.imwrite('samples/random-pix-1.png', img)
    cv2.destroyAllWindows()
