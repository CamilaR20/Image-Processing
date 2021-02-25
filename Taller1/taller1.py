from colorImage import colorImage
import cv2
import numpy as np

if __name__ == '__main__':
    # Pide al usuario la ruta de la imagen y verifica que no sea un objeto vacío (assert)
    # while True:
    #     try:
    #         path_file = input("Ingrese ruta de la imagen: ")
    #         img = colorImage(path_file)
    #         assert np.shape(img.image) != ()
    #         break
    #     except AssertionError:
    #         print("Error: ruta inválida")

    path_file = "/Users/camilaroa/Documents/Camila/PycharmProjects/ProcesamientoImagenes/imgs/lena.png"
    img = colorImage(path_file)

    img.displayProperties()  # Imprime dimensiones de la imagen
    cv2.imshow("Original", img.image)  # Muestra imagen leída
    cv2.waitKey(0)
    cv2.imshow("Gray scale", img.makeGrey())  # Muestra imagen en escala de grises
    cv2.waitKey(0)
    cv2.imshow("Colorized", img.colorizeRGB())  # Muestra versión rojiza de la imagen
    cv2.waitKey(0)
    cv2.imshow("Hue", img.makeHue())  # Muestra versión que resalta los tonos de la imagen
    cv2.waitKey(0)

