import cv2
import numpy as np


class fftClass:
    def __init__(self, im_gray, show=False):
        """Recibe una imagen en grises y guarda en el objeto"""
        # Verifica que la imagen sea cuadrada
        width, height = im_gray.shape
        self.im_size = max(width, height)
        if width == height:
            self.image = im_gray
        elif width > height:
            new_im = np.zeros((width, width), dtype=np.uint8)
            new_im[:, ((width - height) // 2):((width + height) // 2)] = im_gray
            self.image = new_im
        else:
            new_im = np.zeros((height, height), dtype=np.uint8)
            new_im[((height-width) // 2):((height+width) // 2), :] = im_gray
            self.image = new_im

        # Calcular FFT y guardar en el objeto
        self.im_fft = np.fft.fft2(self.image)
        self.fft_shift = np.fft.fftshift(self.im_fft)

        if show:
            # cv2.namedWindow("Grayscale image", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("Grayscale image", 1280, 720)
            cv2.imshow("Grayscale image", self.image)
            cv2.waitKey(0)

    def show(self):
        """Muestra fft de la imagen por 10s"""
        fft_view = np.absolute(self.fft_shift)
        fft_view = np.log(fft_view + np.finfo(np.float32).eps)
        fft_view = fft_view / np.max(fft_view)

        cv2.imshow("Image fft", fft_view)
        cv2.waitKey(0)

    def LPfilter(self, fc=0.5, show=False):
        """Recibe fc (0<fc<1) y muestra fft de la imagen al filtrar
        con un filtro pasa bajas"""
        # create a low pass filter mask
        enum_square = np.linspace(0, self.im_size - 1, self.im_size)
        col_iter, row_iter = np.meshgrid(enum_square, enum_square)
        half_size = (self.im_size / 2 - 1)
        radius_cut_off = int(fc * half_size)
        idx = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) < radius_cut_off
        lp_mask = np.zeros_like(self.image)
        lp_mask[idx] = 1

        # filtering via FFT
        fft_filt = self.fft_shift * lp_mask
        im_filt = np.fft.ifft2(np.fft.fftshift(fft_filt))
        im_filt = np.absolute(im_filt)
        im_filt /= np.max(im_filt)

        if show:
            # show filtered fft
            fft_view = np.absolute(fft_filt)
            fft_view = np.log(fft_view + np.finfo(np.float32).eps)
            fft_view = fft_view / np.max(fft_view)
            cv2.imshow("LP FFT", fft_view)
            cv2.waitKey(0)

            # show filtered image
            cv2.imshow("LP filtered image", im_filt)
            cv2.waitKey(0)

        return im_filt

    def BPfilter(self, f1=0.1, f2=0.6, show=False):
        """Recibe f1 y f2 (0<f1<f2<1) y muestra fft de la imagen al
        filtrar con un filtro pasa banda"""
        enum_square = np.linspace(0, self.im_size - 1, self.im_size)
        col_iter, row_iter = np.meshgrid(enum_square, enum_square)
        half_size = (self.im_size / 2 - 1)
        # create a high pass filter mask
        radius_f1 = int(f1 * half_size)
        idx_hp = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) > radius_f1
        hp_mask = np.zeros_like(self.image)
        hp_mask[idx_hp] = 1
        # create a low pass filter mask
        radius_f2 = int(f2 * half_size)
        idx_lp = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) < radius_f2
        lp_mask = np.zeros_like(self.image)
        lp_mask[idx_lp] = 1
        # Band pass filter mask
        bp_mask = cv2.bitwise_and(lp_mask, hp_mask)

        # filtering via FFT
        fft_filt = self.fft_shift * bp_mask
        im_filt = np.fft.ifft2(np.fft.fftshift(fft_filt))
        im_filt = np.absolute(im_filt)
        im_filt /= np.max(im_filt)

        if show:
            # show filtered fft
            fft_view = np.absolute(fft_filt)
            fft_view = np.log(fft_view + np.finfo(np.float32).eps)
            fft_view = fft_view / np.max(fft_view)
            cv2.imshow("BP FFT", fft_view)
            cv2.waitKey(0)

            # show filtered image
            cv2.imshow("BP filtered image", im_filt)
            cv2.waitKey(0)

        return im_filt


    def HPfilter(self, fc=0.5, show=False):
        """Recibe fc (0<fc<1) y muestra fft de la imagen al
        filtrar con un filtro pasa altas"""
        # create a high pass filter mask
        enum_square = np.linspace(0, self.im_size - 1, self.im_size)
        col_iter, row_iter = np.meshgrid(enum_square, enum_square)
        half_size = (self.im_size / 2 - 1)
        radius_cut_off = int(fc * half_size)
        idx = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) > radius_cut_off
        hp_mask = np.zeros_like(self.image)
        hp_mask[idx] = 1

        # filtering via FFT
        fft_filt = self.fft_shift * hp_mask
        im_filt = np.fft.ifft2(np.fft.fftshift(fft_filt))
        im_filt = np.absolute(im_filt)
        im_filt /= np.max(im_filt)

        if show:
            # show filtered fft
            fft_view = np.absolute(fft_filt)
            fft_view = np.log(fft_view + np.finfo(np.float32).eps)
            fft_view = fft_view / np.max(fft_view)
            cv2.imshow("HP FFT", fft_view)
            cv2.waitKey(0)

            # show filtered image
            cv2.imshow("HP filtered image", im_filt)
            cv2.waitKey(0)

        return im_filt


if __name__ == '__main__':
    im_dog = cv2.imread("imgs/dog.png")
    gray_dog = cv2.cvtColor(im_dog, cv2.COLOR_BGR2GRAY)
    fft_dog = fftClass(gray_dog)
    # fft_dog.show()
    filt_dog = fft_dog.LPfilter(0.1)

    im_cat = cv2.imread("imgs/cat.png")
    gray_cat = cv2.cvtColor(im_cat, cv2.COLOR_BGR2GRAY)
    fft_cat = fftClass(gray_cat)
    # fft_einstein.show()
    filt_cat= fft_cat.HPfilter(0.25)

    new_im = cv2.addWeighted(filt_cat, 0.5, filt_dog, 0.5, 0)

    cv2.imshow("Hybrid", new_im)
    cv2.waitKey(0)

    # img_gray = 255*np.ones((256,512), dtype=np.uint8)
    # im_fft.show()
    # im_fft.LPfilter(0.5)
    # im_fft.HPfilter(0.2)
    # im_fft.BPfilter()

