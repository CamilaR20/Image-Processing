import cv2
import numpy as np
import os

if __name__ == '__main__':
    path_file = os.path.join(os.path.dirname(__file__), 'imgs/lena.png')
    image = cv2.imread(path_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Equalize
    image_gray_eq = cv2.equalizeHist(image_gray)

    # split and merge
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image_hsv)
    s = 255 * np.ones_like(s)
    v = 255 * np.ones_like(v)
    image_hue = cv2.merge((h, s, v))
    image_hue_bgr = cv2.cvtColor(image_hue, cv2.COLOR_HSV2BGR)

    # set 2 last components to 255
    image_hsv[..., 1:] = 255
    image_hue = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("Image", image_hue)
    cv2.waitKey(0)