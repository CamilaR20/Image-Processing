from hough import hough
import cv2
import numpy as np
import os

if __name__ == '__main__':
    path_file = os.path.join(os.path.dirname(__file__), 'imgs/flag3.png')
    image = cv2.imread(path_file)

    high_thresh = 300
    bw_edges = cv2.Canny(image, high_thresh * 0.3, high_thresh, L2gradient=True)
    # _, bw_edges = gradient_map(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    hough = hough(bw_edges)
    accumulator = hough.standard_HT()

    acc_thresh = 50
    N_peaks = 11
    nhood = [25, 9]
    peaks = hough.find_peaks(accumulator, nhood, acc_thresh, N_peaks)

    [_, cols] = image.shape[:2]
    image_draw = np.copy(image)
    for i in range(len(peaks)):
        rho = peaks[i][0]
        theta_ = hough.theta[peaks[i][1]]
        print(theta_)
        theta_pi = np.pi * theta_ / 180
        theta_ = theta_ - 180
        a = np.cos(theta_pi)
        b = np.sin(theta_pi)
        x0 = a * rho + hough.center_x
        y0 = b * rho + hough.center_y
        c = -rho
        x1 = int(round(x0 + cols * (-b)))
        y1 = int(round(y0 + cols * a))
        x2 = int(round(x0 - cols * (-b)))
        y2 = int(round(y0 - cols * a))

        if np.abs(theta_) < 80:
            image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 255], thickness=2)
        elif np.abs(theta_) > 100:
            image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [255, 0, 255], thickness=2)
        else:
            if theta_ > 0:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 0], thickness=2)
            else:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 255], thickness=2)


    # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("frame", 1280, 720)
    # cv2.imshow("frame", bw_edges)
    cv2.namedWindow("lines", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("lines", 1280, 720)
    cv2.imshow("lines", image_draw)
    cv2.waitKey(0)

