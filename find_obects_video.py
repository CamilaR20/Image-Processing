import cv2
import sys
import os

# Find people in video.avi (Background substraction, thresholds,
# morphological operations, contours)
if __name__ == '__main__':
    path_file = os.path.join(os.path.dirname(__file__), 'imgs/vtest.avi')

    # load video
    camera = cv2.VideoCapture(path_file)
    camera.set(cv2.CAP_PROP_POS_FRAMES, 500)

    # properties
    n_frames = camera.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = camera.get(cv2.CAP_PROP_FPS)
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if False:
        backSub = cv2.createBackgroundSubtractorMOG2()
    else:
        backSub = cv2.createBackgroundSubtractorKNN()

    # visualization
    ret = True
    while ret:
        ret, image = camera.read()
        if ret:
            image_draw = image.copy()

            fgMask = backSub.apply(image)
            _, Ibw = cv2.threshold(fgMask, 130, 255, cv2.THRESH_BINARY)
            W = 5
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * W + 1, 2 * W + 1))
            clMask = cv2.morphologyEx(Ibw, cv2.MORPH_CLOSE, kernel)
            opMask = cv2.morphologyEx(clMask, cv2.MORPH_OPEN, kernel)

            contours, hierarchy = cv2.findContours(opMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for idx, cont in enumerate(contours):
                M = cv2.moments(contours[idx])
                area = M['m00']
                if area > 700:
                    x, y, width, height = cv2.boundingRect(contours[idx])
                    cv2.rectangle(image_draw, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # cv2.drawContours(image_draw, contours, -1, (0, 255, 0), 5)

            cv2.imshow("Image", image_draw)
            cv2.imshow('Mask', opMask)
            cv2.waitKey(int(1000 / fps))
