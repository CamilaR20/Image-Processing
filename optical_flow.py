import cv2
import sys
import os
import numpy as np
from enum import Enum

""" Contours (use shapes.png as example)

    python optical_flow.py <path_to_video> <video_name>
"""


class Methods(Enum):
    Simple = 1  # Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
    Farneback = 2
    Deep = 3    # Slow
    PCA = 4
    Sparse = 5
    RLOF = 6


if __name__ == '__main__':
    path = sys.argv[1]
    video_name = sys.argv[2]
    path_file = os.path.join(path, video_name)
    camera = cv2.VideoCapture(path_file)
    flag = True

    # select OF method
    method = Methods.RLOF

    ret = True
    while ret:
        ret, image = camera.read()
        if ret:
            if flag:
                image_now = image.copy()
                flag = False
            else:
                image_past = image_now.copy()
                image_now = image.copy()

                if method == Methods.RLOF:
                    optical_flow = cv2.optflow.createOptFlow_DenseRLOF()
                    flow = optical_flow.calc(image_past, image_now, None)

                else:
                    gray1 = cv2.cvtColor(image_past, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(image_now, cv2.COLOR_BGR2GRAY)

                    if method == Methods.Simple:
                        optical_flow = cv2.optflow.createOptFlow_SimpleFlow()
                    elif method == Methods.Farneback:
                        optical_flow = cv2.optflow.createOptFlow_Farneback()
                    elif method == Methods.Deep:
                        optical_flow = cv2.optflow.createOptFlow_DeepFlow()
                    elif method == Methods.PCA:
                        optical_flow = cv2.optflow.createOptFlow_PCAFlow()
                    elif method == Methods.Sparse:
                        optical_flow = cv2.optflow.createOptFlow_SparseToDense()

                    flow = optical_flow.calc(gray1, gray2, None)

                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv = np.zeros_like(image)
                hsv[..., 1] = 255
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                cv2.imshow("Image", rgb)
                cv2.waitKey(1)
