import cv2
import os
import json
import numpy as np
from camera_model import *

if __name__ == '__main__':
    # Read json files with intrinsic parameters of the camera and the position
    json_file = os.path.join(os.path.dirname(__file__), 'camera_parameters1.json')

    with open(json_file) as fp:
        json_data = json.load(fp)

    K, d, h, tilt, pan = json_data.values()
    K = np.array(K)

    # Dimensions of the image given the camera 768×1024: replace with appropriate vakues
    width = 768
    height = 1024

    # Extrinsic parameters for the camera
    R = set_rotation(tilt, pan, 0)
    t = np.array([0, -d, h])

    # Create camerea model
    camera = ProjectiveCamera(K, width, height, R, t)

    # Draw projected cube
    side = 1
    cube_3D = np.array([[0, 0, 0], [side, 0, 0], [side, side, 0], [0, side, 0],
                        [0, side, side], [side, side, side], [side, 0, side], [0, 0, side]])
    cube_2D = projective_camera_project(cube_3D, camera)

    image_projective = 255 * np.ones(shape=[camera.height, camera.width, 3], dtype=np.uint8)
    cv2.line(image_projective, (cube_2D[0][0], cube_2D[0][1]), (cube_2D[1][0], cube_2D[1][1]), (255, 0, 0), 3)
    cv2.line(image_projective, (cube_2D[0][0], cube_2D[0][1]), (cube_2D[3][0], cube_2D[3][1]), (0, 255, 0), 3)

    cv2.line(image_projective, (cube_2D[1][0], cube_2D[1][1]), (cube_2D[6][0], cube_2D[6][1]), (255, 0, 0), 3)
    cv2.line(image_projective, (cube_2D[1][0], cube_2D[1][1]), (cube_2D[2][0], cube_2D[2][1]), (0, 255, 0), 3)

    cv2.line(image_projective, (cube_2D[2][0], cube_2D[2][1]), (cube_2D[3][0], cube_2D[3][1]), (0, 0, 255), 3)
    cv2.line(image_projective, (cube_2D[2][0], cube_2D[2][1]), (cube_2D[5][0], cube_2D[5][1]), (0, 0, 255), 3)

    cv2.line(image_projective, (cube_2D[3][0], cube_2D[3][1]), (cube_2D[4][0], cube_2D[4][1]), (0, 0, 255), 3)

    cv2.line(image_projective, (cube_2D[4][0], cube_2D[4][1]), (cube_2D[5][0], cube_2D[5][1]), (0, 0, 255), 3)
    cv2.line(image_projective, (cube_2D[4][0], cube_2D[4][1]), (cube_2D[7][0], cube_2D[7][1]), (0, 255, 0), 3)

    cv2.line(image_projective, (cube_2D[6][0], cube_2D[6][1]), (cube_2D[5][0], cube_2D[5][1]), (0, 255, 0), 3)
    cv2.line(image_projective, (cube_2D[6][0], cube_2D[6][1]), (cube_2D[7][0], cube_2D[7][1]), (255, 0, 0), 3)

    cv2.line(image_projective, (cube_2D[0][0], cube_2D[0][1]), (cube_2D[7][0], cube_2D[7][1]), (255, 0, 0), 3)

    cv2.imshow("Image", image_projective)
    cv2.waitKey(0)







