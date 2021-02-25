import cv2
import sys
import os
import numpy as np

points = []


def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image_draw = np.copy(image)

    points1 = []
    points2 = []

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click)

    point_counter = 0
    while True:
        cv2.imshow("Image", image_draw)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("x"):
            points1 = points.copy()
            points = []
            break
        if len(points) > point_counter:
            point_counter = len(points)
            cv2.circle(image_draw, (points[-1][0], points[-1][1]), 3, [0, 0, 255], -1)

    point_counter = 0
    while True:
        cv2.imshow("Image", image_draw)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("x"):
            points2 = points.copy()
            points = []
            break
        if len(points) > point_counter:
            point_counter = len(points)
            cv2.circle(image_draw, (points[-1][0], points[-1][1]), 3, [255, 0, 0], -1)

    N = min(len(points1), len(points2))
    assert N >= 4, 'At least four points are required'

    pts1 = np.array(points1[:N])
    pts2 = np.array(points2[:N])
    if False:
        H, _ = cv2.findHomography(pts1, pts2, method=0)
    else:
        H, _ = cv2.findHomography(pts1, pts2, method=cv2.RANSAC)

    image_warped = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))
    cv2.imshow("Image", image)
    cv2.imshow("Image warped", image_warped)
    cv2.waitKey(0)