import cv2
import numpy as np
import os

if __name__ == '__main__':
    image_1 = cv2.imread(os.path.join(os.path.dirname(__file__), 'imgs/cat.png'))
    image_2 = cv2.imread(os.path.join(os.path.dirname(__file__), 'imgs/dog.png'))
    image_gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    image_gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    # Low-pass via Gaussian filtering
    N = 21
    image_filtered_lp = cv2.GaussianBlur(image_gray_1, (N, N), 2.5, 2.5)

    # High-pass via FFT
    image_gray_fft_2 = np.fft.fft2(image_gray_2)
    image_gray_fft_shift_2 = np.fft.fftshift(image_gray_fft_2)

    num_rows, num_cols = (image_gray_1.shape[0], image_gray_1.shape[1])
    enum_rows = np.linspace(0, num_rows - 1, num_rows)
    enum_cols = np.linspace(0, num_cols - 1, num_cols)
    col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
    high_pass_mask = np.zeros_like(image_gray_1)
    freq_cut_off_hp = 0.005  # it should less than 1
    half_size = num_rows / 2 - 1  # here we assume num_rows = num_columns
    radius_cut_off_hp = int(freq_cut_off_hp * half_size)
    idx_hp = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) > radius_cut_off_hp
    high_pass_mask[idx_hp] = 1

    fft_filtered_hp = image_gray_fft_shift_2 * high_pass_mask
    image_filtered_hp = np.fft.ifft2(np.fft.fftshift(fft_filtered_hp))
    image_filtered_hp = np.absolute(image_filtered_hp)
    image_filtered_hp = image_filtered_hp.astype(np.uint8)

    # hybrid image
    image_hybrid = cv2.addWeighted(image_filtered_lp, 0.5, image_filtered_hp, 0.5, 0)

    # visualize images
    cv2.imshow("lp", image_filtered_lp)
    cv2.imshow("hp", image_filtered_hp)
    cv2.imshow("hybrid", image_hybrid)

    # save image
    cv2.imwrite("hybrid.png", image_hybrid)
    cv2.waitKey(0)