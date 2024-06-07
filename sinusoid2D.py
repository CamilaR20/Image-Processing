import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # sinusoid generation
    num_rows, num_cols = (512, 512)
    enum_rows = np.linspace(0, num_rows - 1, num_rows)
    enum_cols = np.linspace(0, num_cols - 1, num_cols)
    col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
    u, v = (10, 10)
    omega = np.sqrt(u**2 + v**2)
    sinusoid = np.sin(2 * np.pi * (u * row_iter / num_rows + v * col_iter / num_cols))

    # sinusoid visualization
    sinusoid_view = (sinusoid.copy() + 1) / 2
    cv2.imshow("sinusoid", sinusoid_view)
    cv2.waitKey(0)

    sinusoid_fft = np.fft.fft2(sinusoid)
    fft_shift = np.fft.fftshift(sinusoid_fft)

    # fft visualization
    fft_mag = np.abs(fft_shift)
    fft_view = np.log(fft_mag + 1)
    fft_view = fft_view / np.max(fft_view)

    cv2.imshow("sinusoid fft", fft_view)
    cv2.waitKey(0)

    plt.imshow(fft_view, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
