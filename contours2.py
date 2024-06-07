import cv2
import os
import numpy as np

if __name__ == '__main__':
    path_file = path_file = os.path.join(os.path.dirname(__file__), 'imgs/coins.png')
    image = cv2.imread(path_file)
    image_draw = image.copy()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, Ibw_coins = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # floodfill
    height, width = image.shape[:2]
    mask = np.zeros((height + 2, width + 2), np.uint8)
    coins_floodfill = Ibw_coins.copy()
    cv2.floodFill(coins_floodfill, mask, (0, 0), 255)
    coins_floodfill_inv = cv2.bitwise_not(coins_floodfill)

    # coins mask
    coins = cv2.bitwise_or(Ibw_coins, coins_floodfill_inv)

    # countours
    contours, hierarchy = cv2.findContours(coins, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_draw = image.copy()
    for idx, cont in enumerate(contours):
        if len(contours[idx]) > 20:
            hull = cv2.convexHull(contours[idx])
            cv2.drawContours(image_draw, contours, idx, (0, 255, 255), 2)
            cv2.drawContours(image_draw, [hull], 0, (255, 0, 0), 2)
            M = cv2.moments(contours[idx])
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            area = M['m00']
            x, y, width, height = cv2.boundingRect(contours[idx])
            cv2.rectangle(image_draw, (x, y), (x + width, y + height), (0, 0, 255), 2)
            (x, y), radius = cv2.minEnclosingCircle(contours[idx])
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(image_draw, center, radius, (0, 255, 0), 2)

    # cv2.imshow("Image", image)
    # cv2.imshow("Ibw", Ibw_coins)
    # cv2.imshow("FF", coins_floodfill_inv)
    # cv2.imshow("mask", coins)
    cv2.imshow("Contours", image_draw)
    cv2.waitKey(0)
