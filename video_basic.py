import cv2
import sys
import os

if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)

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
