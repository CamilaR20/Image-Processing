import cv2
import numpy as np
import sys
import os

if __name__ == '__main__':
    path_file = os.path.join(os.path.dirname(__file__), 'imgs/lena.png')
    image = cv2.imread(path_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_draw = np.copy(image)

    # Harris
    dst = cv2.cornerHarris(image_gray.astype(np.float32), 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    image_draw[dst > 0.01 * dst.max()] = [0, 0, 255]

    # Shi-Tomasi
    corners = cv2.goodFeaturesToTrack(image_gray, 100, 0.0001, 10)
    corners = corners.astype(int)
    for i in corners:
        x, y = i.ravel()
        cv2.circle(image_draw, (x, y), 3, [255, 0, 0], -1)

    # sift and orb
    # sift = cv2.SIFT_create(nfeatures=100)
    orb = cv2.ORB_create(nfeatures=100)

    # keypoints_sift, descriptors = sift.detectAndCompute(image_gray, None)
    keypoints_orb, descriptors = orb.detectAndCompute(image_gray, None)
    image_draw = cv2.drawKeypoints(image_gray, keypoints_orb, None)

    cv2.imshow("Image", image_draw)
    cv2.waitKey(0)
