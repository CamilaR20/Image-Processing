import cv2
import sys
import os

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
    pos = camera.get(cv2.CAP_PROP_POS_FRAMES)
    bitrate = camera.get(cv2.CAP_PROP_BITRATE)

    # visualization
    ret = True
    while ret:
        ret, image = camera.read()
        if ret:
            cv2.imshow("Image", image)
            cv2.waitKey(int(1000 / fps))
