import cv2

if __name__ == '__main__':
    path_file = "imgs/soccer.png"
    image = cv2.imread(path_file)

    # Hue histogram
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_hsv = cv2.calcHist([image_hsv], [0], None, [180], [0, 180])

    # Hue histogram max and location of max
    max_val = hist_hsv.max()
    max_pos = int(hist_hsv.argmax())

    # grass mask
    lim_inf = (max_pos - 10, 0, 0)
    lim_sup = (max_pos + 10, 255, 255)
    mask_grass = cv2.inRange(image_hsv, lim_inf, lim_sup)
    mask_not_grass = cv2.bitwise_not(mask_grass)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 1280, 720)
    cv2.imshow("Image", mask_not_grass)
    cv2.waitKey(0)