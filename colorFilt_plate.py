import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path_file = "imgs/placa5.png"
    image = cv2.imread(path_file)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

    I_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # hist_Cb = cv2.calcHist([I_YCrCb], [2], None, [256], [0, 256])
    # plt.plot(hist_Cb)
    # plt.xlim([0, 256])
    # plt.show()

    # Otsu's global threshold
    ret, Ibw_Cb = cv2.threshold(I_YCrCb[:, :, 2], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imshow("Mask Cb", Ibw_Cb)
    # cv2.waitKey(0)

    # Hue histogram, location of max, color filtering
    I_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_hsv = cv2.calcHist([I_hsv], [0], Ibw_Cb, [180], [0, 180])
    # plt.plot(hist_hsv)
    # plt.xlim([0, 180])
    # plt.show()

    max_pos = int(hist_hsv.argmax())
    lim_inf = (max_pos - 10, 0, 0)
    lim_sup = (max_pos + 10, 255, 255)
    Ibw_H = cv2.inRange(I_hsv, lim_inf, lim_sup)
    # cv2.imshow("Mask H", Ibw_H)
    # cv2.waitKey(0)

    ret, Ibw_sat = cv2.threshold(I_hsv[:, :, 1], 128, 255, cv2.THRESH_BINARY)

    # Mask
    Ibw_mask = cv2.bitwise_and(Ibw_H, Ibw_sat)
    cv2.imshow("Mask", Ibw_mask)
    cv2.waitKey(0)

    # Erosion and dilation
    W = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*W+1, 2*W+1))  # Objeto de opencv para la ventana
    mask_eroded = cv2.morphologyEx(Ibw_mask, cv2.MORPH_ERODE, kernel)
    mask_dilated = cv2.morphologyEx(Ibw_mask, cv2.MORPH_DILATE, kernel)

    cv2.imshow("Eroded", mask_eroded)
    cv2.waitKey(0)

    cv2.imshow("Dilated", mask_dilated)
    cv2.waitKey(0)
