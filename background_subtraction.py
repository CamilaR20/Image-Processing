import cv2
import sys
import os

if __name__ == '__main__':
    path_file = os.path.join(os.path.dirname(__file__), 'imgs/vtest.avi')

    if True:
        backSub = cv2.createBackgroundSubtractorMOG2()
    else:
        backSub = cv2.createBackgroundSubtractorKNN()

    camera = cv2.VideoCapture(path_file)
    ret = True

    while ret:
        ret, image = camera.read()
        if ret:
            fgMask = backSub.apply(image)
            cv2.imshow("Image", image)
            cv2.imshow('FG Mask', fgMask)
            cv2.waitKey(100)
