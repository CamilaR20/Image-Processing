import cv2
import numpy as np
from noise import noise
import os

if __name__ == '__main__':
    image = cv2.imread(os.path.join(os.path.dirname(__file__), 'imgs/lena.png'))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # add noise
    image_gray_noisy = noise("s&p", image_gray.astype(float) / 255)
    image_gray_noisy = (255 * image_gray_noisy).astype(np.uint8)

    # window size
    N = 7

    # 1) convolution (kernel, mascara, filtro de convolucion, respuesta impulso)
    kernel = np.array([[0, 0.5, 0], [0.5, 2, 0.5], [0, 0.5, 0]]) / 3
    image_convolved = cv2.filter2D(image_gray_noisy, -1, kernel)

    # 2) averaging
    image_blur = cv2.blur(image_gray_noisy, (N, N))

    # 3) gaussian low-pass
    image_gauss_lp = cv2.GaussianBlur(image_gray_noisy, (N, N), 1.5, 1.5)

    # 4) gaussian high-pass
    kernel_gauss = cv2.getGaussianKernel(N, 2.5)
    kernel_gauss_2D = np.multiply(kernel_gauss.T, kernel_gauss)
    impulse_2D = np.zeros((N, N), dtype=kernel_gauss.dtype)
    NN = int((N - 1) / 2)
    impulse_2D[NN, NN] = 1
    kernel_gauss_2D_hp = impulse_2D - kernel_gauss_2D
    image_gauss_hp = cv2.filter2D(image_gray, -1, kernel_gauss_2D_hp)

    # 5) gaussian band-pass
    image_gauss_1 = cv2.GaussianBlur(image_gray, (N, N), 0.5, 0.5)
    image_gauss_2 = cv2.GaussianBlur(image_gray, (N, N), 2.5, 2.5)
    image_gauss_bp = (np.abs(image_gauss_2.astype(float) - image_gauss_1.astype(float))).astype(np.uint8)

    cv2.imshow("Filtered image", image_gauss_bp)
    cv2.waitKey(0)
