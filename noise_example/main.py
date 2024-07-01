import cv2
import numpy as np
from noise import noise
import os
import time


def sqrt_mse(im1, im2):
    difference = (im1.astype(float) - im2.astype(float)) ** 2
    error = (1 / (im1.shape[0] * im1.shape[1])) * np.sum(difference)
    return np.sqrt(error)


if __name__ == '__main__':
    image = cv2.imread(os.path.join(os.path.dirname(__file__), '../imgs/lena.png'))
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Image with s&p noise
    im_sp_noisy = noise("s&p", im_gray.astype(float) / 255)
    im_sp_noisy = (255 * im_sp_noisy).astype(np.uint8)

    # Image with gaussian noise
    im_gauss_noisy = noise("gauss", im_gray.astype(float) / 255)
    im_gauss_noisy = (255 * im_gauss_noisy).astype(np.uint8)

    noisy_imgs = [["noisy_s&p", im_sp_noisy], ["noisy_gauss", im_gauss_noisy]]

    # Folder to save results
    path = os.path.dirname(__file__)
    ext = ".png"

    # Save original image
    im_name = "original" + ext
    path_file = os.path.join(path, im_name)
    cv2.imwrite(path_file, im_gray)

    for noisy_im in noisy_imgs:
        # Save noisy image
        im_name = noisy_im[0] + ext
        cv2.imwrite(os.path.join(path, im_name), noisy_im[1])

        N = 7
        # ------------Gaussian filter 7x7 con sigma=1.5---------------
        start_time = time.time()
        im_filt_gauss = cv2.GaussianBlur(noisy_im[1], (N, N), 1.5, 1.5)
        print("Timing of gaussian filter (s)= ", (time.time() - start_time))
        # Save filtered image
        im_name = noisy_im[0] + "_filt_gauss" + ext
        cv2.imwrite(os.path.join(path, im_name), im_filt_gauss)
        # Noise estimate
        im_noise = abs(noisy_im[1] - im_filt_gauss)
        # Save noise estimate
        im_name = noisy_im[0] + "_est_gauss" + ext
        cv2.imwrite(os.path.join(path, im_name), im_noise)
        # Compute RMSE after gaussian filter
        print("MSE after gaussian filter", noisy_im[0], sqrt_mse(im_gray, im_filt_gauss), "\n")

        # # ---------------------Median filter 7x7----------------------
        start_time = time.time()
        im_filt_median = cv2.medianBlur(noisy_im[1], N)
        print("Timing of median filter (s)= ", (time.time() - start_time))
        # Save filtered image
        im_name = noisy_im[0] + "_filt_median" + ext
        cv2.imwrite(os.path.join(path, im_name), im_filt_median)
        # Estimate noise
        im_noise = abs(noisy_im[1] - im_filt_median)
        # Save noise estimate
        im_name = noisy_im[0] + "_est_median" + ext
        cv2.imwrite(os.path.join(path, im_name), im_noise)
        # Compute RMSE afeter median filter
        print("RMSE after median filter", noisy_im[0], sqrt_mse(im_gray, im_filt_median), "\n")

        # ---Bilateral filter with d=15, sigmaColor = sigmaSpace = 25---
        start_time = time.time()
        im_filt_bil = cv2.bilateralFilter(noisy_im[1], 15, 25, 25)
        print("Timing of bilateral filter (s)= ", (time.time() - start_time))
        # Save filtered image
        im_name = noisy_im[0] + "_filt_bilateral" + ext
        cv2.imwrite(os.path.join(path, im_name), im_filt_bil)
        # Noise estimagte
        im_noise = abs(noisy_im[1] - im_filt_bil)
        im_name = noisy_im[0] + "_est_bilateral" + ext
        cv2.imwrite(os.path.join(path, im_name), im_noise)
        # Compute RMSE
        print("RMSE after bilateral filter", noisy_im[0], sqrt_mse(im_gray, im_filt_bil), "\n")

        # -----NLM filter with h=5, windowSize = 15, searchSize = 25-----
        start_time = time.time()
        im_filt_nlm = cv2.fastNlMeansDenoising(noisy_im[1], 5, 15, 25)
        print("Timing of NLM filter (s)= ", (time.time() - start_time))
        # Save filtered image
        im_name = noisy_im[0] + "_filt_NLM" + ext
        cv2.imwrite(os.path.join(path, im_name), im_filt_nlm)
        # Noise estimate
        im_noise = abs(noisy_im[1] - im_filt_nlm)
        im_name = noisy_im[0] + "_est_NLM" + ext
        cv2.imwrite(os.path.join(path, im_name), im_noise)
        # Compute RMSE
        print("RMSE after NLM filter", noisy_im[0], sqrt_mse(im_gray, im_filt_nlm), "\n")
