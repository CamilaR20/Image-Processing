import numpy as np
import cv2
import glob
import os
import json

# Parámetros del tablero
grid_w = 9
grid_h = 6
square_size = 0.025

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((grid_h * grid_w, 3), np.float32)
objp[:, :2] = np.mgrid[0:grid_w, 0:grid_h].T.reshape(-1, 2)
objp = objp * square_size

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

path = '/Users/camilaroa/Documents/Camila/PycharmProjects/ProcesamientoImagenes/calib_iphone'
path_file = os.path.join(path, '*.png')

images = glob.glob(path_file)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (grid_w, grid_h), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (grid_w, grid_h), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(250)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error / len(objpoints)))


file_name = 'calibration.json'
json_file = os.path.join(path, file_name)

data = {
    'K': mtx.tolist(),
    'distortion': dist.tolist()
}

with open(json_file, 'w') as fp:
    json.dump(data, fp, sort_keys=True, indent=1, ensure_ascii=False)

