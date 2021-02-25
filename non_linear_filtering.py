import cv2
import numpy as np
from noise import noise
import os

if __name__ == '__main__':
    image = cv2.imread("imgs/lena.png")
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # add noise
    image_gray_noisy = noise("poisson", image_gray.astype(np.float) / 255)
    image_gray_noisy = (255 * image_gray_noisy).astype(np.uint8)

    # median
    image_median = cv2.medianBlur(image_gray_noisy, 5)

    # bilateral
    image_bilateral = cv2.bilateralFilter(image_gray_noisy, 15, 25, 25)

    # nlm
    image_nlm = cv2.fastNlMeansDenoising(image_gray_noisy, 5, 15, 25)

    cv2.imshow("Image", image_nlm)
    cv2.waitKey(0)
