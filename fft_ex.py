import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    path_file = "imgs/lena.png"
    image = cv2.imread(path_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray_fft = np.fft.fft2(image_gray)
    image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

    # fft visualization
    image_gray_fft_mag = np.absolute(image_gray_fft_shift)
    image_fft_view = np.log(image_gray_fft_mag + np.finfo(np.float32).eps)
    image_fft_view = image_fft_view / np.max(image_fft_view)

    # create a low pass filter mask
    num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
    enum_rows = np.linspace(0, num_rows - 1, num_rows)
    enum_cols = np.linspace(0, num_cols - 1, num_cols)
    col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
    freq_cut_off = 0.4  # it should less than 1
    half_size = num_rows / 2 - 1  # here we assume num_rows = num_columns
    radius_cut_off = int(freq_cut_off * half_size)
    idx = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) < radius_cut_off
    low_pass_mask = np.zeros_like(image_gray)
    low_pass_mask[idx] = 1

    # filtering via FFT
    fft_filtered = image_gray_fft_shift * low_pass_mask
    image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
    image_filtered = np.absolute(image_filtered)
    image_filtered /= np.max(image_filtered)

    cv2.imshow("Image", image_filtered)
    cv2.waitKey(0)
