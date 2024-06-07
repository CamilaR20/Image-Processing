from colorImage import colorImage
import cv2
import numpy as np
import os

if __name__ == '__main__':
    # Asks path of an image from the user and checks it is not empty (assert)
    # while True:
    #     try:
    #         path_file = input("Type image path: ")
    #         img = colorImage(path_file)
    #         assert np.shape(img.image) != ()
    #         break
    #     except AssertionError:
    #         print("Error: path is not valid")

    path_file = os.path.join(os.path.dirname(__file__), '../imgs/lena.png')
    img = colorImage(path_file)

    img.displayProperties()  # Prints dimensions of the image
    cv2.imshow("Original", img.image)  # Shows image from path
    cv2.waitKey(0)
    cv2.imshow("Gray scale", img.makeGrey())  # Shows grayscale image
    cv2.waitKey(0)
    cv2.imshow("Colorized", img.colorizeRGB())  # Shows reddish image
    cv2.waitKey(0)
    cv2.imshow("Hue", img.makeHue())  # Shows version that highlights the hue of the image
    cv2.waitKey(0)

