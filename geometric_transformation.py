import cv2
import numpy as np
import sys
import os

if __name__ == '__main__':
    path_file = os.path.join(os.path.dirname(__file__), 'imgs/lena.png')
    image = cv2.imread(path_file)

    # resizing
    image_resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)

    # translation
    tx = 50
    ty = 20
    M_t = np.float32([[1, 0, tx], [0, 1, ty]])
    image_translation = cv2.warpAffine(image, M_t, (image.shape[1], image.shape[0]))

    # scaling
    sx = 0.75
    sy = 1.25
    M_s = np.float32([[sx, 0, 0], [0, sy, 0]])
    image_scale = cv2.warpAffine(image, M_s, (700, 700))

    # rotation
    theta = 5
    theta_rad = theta * np.pi / 180
    # M_rot = np.float32([[np.cos(theta_rad), -np.sin(theta_rad), 0],
    #                    [np.sin(theta_rad), np.cos(theta_rad), 0]])
    cx = image.shape[1] / 2
    cy = image.shape[0] / 2
    M_rot = cv2.getRotationMatrix2D((cx, cy), theta, 1)
    image_rotation = cv2.warpAffine(image, M_rot, image.shape[:2])

    # similarity
    M_sim = np.float32([[sx * np.cos(theta_rad), -np.sin(theta_rad), tx],
                        [np.sin(theta_rad), sy * np.cos(theta_rad), ty]])
    image_similarity = cv2.warpAffine(image, M_sim, image.shape[:2])

    # affine
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M_affine = cv2.getAffineTransform(pts1, pts2)
    image_affine = cv2.warpAffine(image, M_affine, image.shape[:2])

    cv2.imshow("Image after affine transformation", image_affine)
    cv2.waitKey(0)
