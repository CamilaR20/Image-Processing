import cv2
from camera_model import *

if __name__ == '__main__':

    fx = 1000
    fy = 1000
    cx = 640
    cy = 360
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    width = 1280
    height = 720
    camera = PinholeCamera(K, width, height)

    p_3D = np.array([[1.0, 1.0, 3.0]])
    p_2D = pinhole_camera_project(p_3D, camera)

    dist = 5.0
    square_3D = np.array([[0.5, 0.5, dist], [0.5, -0.5, dist], [-0.5, -0.5, dist], [-0.5, 0.5, dist]])
    square_2D = pinhole_camera_project(square_3D, camera)

    image_pinhole = 255 * np.ones(shape=[camera.height, camera.width, 3], dtype=np.uint8)
    cv2.line(image_pinhole, (square_2D[0][0], square_2D[0][1]), (square_2D[1][0], square_2D[1][1]), (200, 1, 255), 3)
    cv2.line(image_pinhole, (square_2D[1][0], square_2D[1][1]), (square_2D[2][0], square_2D[2][1]), (200, 1, 255), 3)
    cv2.line(image_pinhole, (square_2D[2][0], square_2D[2][1]), (square_2D[3][0], square_2D[3][1]), (200, 1, 255), 3)
    cv2.line(image_pinhole, (square_2D[3][0], square_2D[3][1]), (square_2D[0][0], square_2D[0][1]), (200, 1, 255), 3)

    cv2.imshow("Image", image_pinhole)
    cv2.waitKey(0)

