import cv2
import numpy as np


class colorImage:
    def __init__(self, path_file):
        """Reads image from path and saves it"""
        self.image = cv2.imread(path_file)

    def displayProperties(self):
        """Prints dimensions of the image"""
        im_dim = self.image.shape
        print("Size of the image: %d x %d" % (im_dim[0], im_dim[1]))

    def makeGrey(self):
        """Returns a grayscale version fo the image"""
        im_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return im_gray

    def colorizeRGB(self, color='red'):
        """Returns a colorized version of the image depending on the color argument:
        color = 'red' -> returns a reddish version
        color = 'green' -> returns a greenish version
        color = 'blue' -> returns a blueish version
        """
        # Para conocer el número del canal que indica el argumento 'color' del método
        color_dict = {'blue': 0, 'green': 1, 'red': 2}

        im_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Crear copia de la imagen, poneer todos los canales en 0
        im_colored = self.image.copy()
        im_colored[:, :, :] = 0
        # Copiar la imagen en escala de grises al canal indicado por el argumento 'color'
        im_colored[:, :, color_dict[color]] = im_gray
        return im_colored

    def makeHue(self):
        """Returns version that highlihgts the hue of the image."""
        im_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        im_hsv[:, :, [1, 2]] = 255
        im_hue = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return im_hue
