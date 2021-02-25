import cv2
import numpy as np


class colorImage:
    def __init__(self, path_file):
        """Leer imagen de una ruta específica y guardar en el objeto"""
        self.image = cv2.imread(path_file)

    def displayProperties(self):
        """Imprime dimensiones de la imagen"""
        im_dim = self.image.shape
        print("Tamaño de la imágen: %d x %d" % (im_dim[0], im_dim[1]))

    def makeGrey(self):
        """Devuelve una versión en escala de grises de la imagen"""
        im_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return im_gray

    def colorizeRGB(self, color='red'):
        """Devuelve una versión colorizada de la imagen dependiendo del argumento color:
        color = 'red' -> devuelve versión rojiza
        color = 'green' -> devuelve versión verdoza
        color = 'blue' -> devuelve versión azuloza
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
        """Devuelve versión que resalta los tonos de la imagen"""
        im_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        im_hsv[:, :, [1, 2]] = 255
        im_hue = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return im_hue
