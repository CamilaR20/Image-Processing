import cv2
import os

if __name__ == '__main__':
    path_file = os.path.join(os.path.dirname(__file__), 'imgs/shapes.png')
    image = cv2.imread(path_file)
    image_draw = image.copy()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, Ibw_shapes = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(Ibw_shapes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    color = (0, 0, 0)
    cv2.drawContours(image_draw, contours, -1, color, 5)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 1280, 720)
    cv2.imshow("Image", image_draw)
    cv2.waitKey(0)