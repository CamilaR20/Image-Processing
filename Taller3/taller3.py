import cv2
import numpy as np
from noise import noise
import os
import time


def sqrt_mse(im1, im2):
    difference = (im1.astype(np.float) - im2.astype(np.float)) ** 2
    error = (1 / (im1.shape[0] * im1.shape[1])) * np.sum(difference)
    return np.sqrt(error)


if __name__ == '__main__':
    image = cv2.imread("imgs/lena.png")
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Imagen con ruido s&p
    im_sp_noisy = noise("s&p", im_gray.astype(np.float) / 255)
    im_sp_noisy = (255 * im_sp_noisy).astype(np.uint8)

    # Imagen con ruido gaussiano
    im_gauss_noisy = noise("gauss", im_gray.astype(np.float) / 255)
    im_gauss_noisy = (255 * im_gauss_noisy).astype(np.uint8)

    noisy_imgs = [["noisy_s&p", im_sp_noisy], ["noisy_gauss", im_gauss_noisy]]

    # Carpeta para guardar imágenes y extensión de las imágenes
    path = "/Users/camilaroa/Documents/Camila/PycharmProjects/ProcesamientoImagenes/results"
    ext = ".png"

    # Guardar imagen original
    im_name = "original" + ext
    path_file = os.path.join(path, im_name)
    cv2.imwrite(path_file, im_gray)

    for noisy_im in noisy_imgs:
        # Guardar imagen con ruido
        im_name = noisy_im[0] + ext
        cv2.imwrite(os.path.join(path, im_name), noisy_im[1])

        N = 7
        # ------------Filtro gaussiano 7x7 con sigma=1.5---------------
        start_time = time.time()
        im_filt_gauss = cv2.GaussianBlur(noisy_im[1], (N, N), 1.5, 1.5)
        print("Tiempo filtro gaussiano (s)= ", (time.time() - start_time))
        # Guardar imagen filtrada
        im_name = noisy_im[0] + "_filt_gauss" + ext
        cv2.imwrite(os.path.join(path, im_name), im_filt_gauss)
        # Estimación del ruido con filtro gaussiano
        im_noise = abs(noisy_im[1] - im_filt_gauss)
        # Guardar imagen de estimación de ruido
        im_name = noisy_im[0] + "_est_gauss" + ext
        cv2.imwrite(os.path.join(path, im_name), im_noise)
        # Calcular raíz cuadrada del MSE
        print("MSE filtro gaussiano en", noisy_im[0], sqrt_mse(im_gray, im_filt_gauss), "\n")

        # # ---------------------Filtro mediana 7x7----------------------
        start_time = time.time()
        im_filt_median = cv2.medianBlur(noisy_im[1], N)
        print("Tiempo filtro mediana (s)= ", (time.time() - start_time))
        # Guardar imagen filtrada
        im_name = noisy_im[0] + "_filt_median" + ext
        cv2.imwrite(os.path.join(path, im_name), im_filt_median)
        # Estimación del ruido
        im_noise = abs(noisy_im[1] - im_filt_median)
        # Guardar imagen de estimación de ruido
        im_name = noisy_im[0] + "_est_median" + ext
        cv2.imwrite(os.path.join(path, im_name), im_noise)
        # Calcular raíz cuadrada del MSE
        print("MSE filtro mediana en", noisy_im[0], sqrt_mse(im_gray, im_filt_median), "\n")

        # ---Filtro bilateral con d=15, sigmaColor = sigmaSpace = 25---
        start_time = time.time()
        im_filt_bil = cv2.bilateralFilter(noisy_im[1], 15, 25, 25)
        print("Tiempo filtro bilateral (s)= ", (time.time() - start_time))
        # Guardar imagen filtrada
        im_name = noisy_im[0] + "_filt_bilateral" + ext
        cv2.imwrite(os.path.join(path, im_name), im_filt_bil)
        # Estimación del ruido
        im_noise = abs(noisy_im[1] - im_filt_bil)
        im_name = noisy_im[0] + "_est_bilateral" + ext
        cv2.imwrite(os.path.join(path, im_name), im_noise)
        # Calcular raíz cuadrada del MSE
        print("MSE filtro bilateral en", noisy_im[0], sqrt_mse(im_gray, im_filt_bil), "\n")

        # -----Filtro NLM con h=5, windowSize = 15, searchSize = 25-----
        start_time = time.time()
        im_filt_nlm = cv2.fastNlMeansDenoising(noisy_im[1], 5, 15, 25)
        print("Tiempo filtro NLM (s)= ", (time.time() - start_time))
        # Guardar imagen filtrada
        im_name = noisy_im[0] + "_filt_NLM" + ext
        cv2.imwrite(os.path.join(path, im_name), im_filt_nlm)
        # Estimación del ruido
        im_noise = abs(noisy_im[1] - im_filt_nlm)
        im_name = noisy_im[0] + "_est_NLM" + ext
        cv2.imwrite(os.path.join(path, im_name), im_noise)
        # Calcular raíz cuadrada del MSE
        print("MSE filtro NLM en", noisy_im[0], sqrt_mse(im_gray, im_filt_nlm), "\n")
