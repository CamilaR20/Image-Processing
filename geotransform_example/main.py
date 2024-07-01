import cv2
import sys
import numpy as np
import os

class geomTransform:
    def __init__(self, pts1, pts2):
        self.pts1 = pts1
        self.pts2 = pts2
        self.M_affine = cv2.getAffineTransform(pts1, pts2)

    def affineTransform(self, image):
        # Compute affine transform of image and show it
        image_affine = cv2.warpAffine(image, self.M_affine, image.shape[:2])
        cv2.imshow("Affine transform of image", image_affine)
        cv2.waitKey(0)

    def estSimilarity(self):
        # Estimate rotation, translation and scale parameters from affine transform matrix
        s0 = np.sqrt(self.M_affine[0, 0] ** 2 + self.M_affine[1, 0] ** 2)
        s1 = np.sqrt(self.M_affine[0, 1] ** 2 + self.M_affine[1, 1] ** 2)
        theta = -np.arctan(self.M_affine[1, 0] / self.M_affine[0, 0])
        x0 = (self.M_affine[0, 2] * np.cos(theta) - self.M_affine[1, 2] * np.sin(theta)) / s0
        x1 = (self.M_affine[0, 2] * np.sin(theta) + self.M_affine[1, 2] * np.cos(theta)) / s1

        self.M_sim = np.float32(
            [[s0 * np.cos(theta), s1 * np.sin(theta), (s0 * x0 * np.cos(theta) + s1 * x1 * np.sin(theta))],
             [-s0 * np.sin(theta), s1 * np.cos(theta), (s1 * x1 * np.cos(theta) - s0 * x0 * np.sin(theta))]])

    def similarityTransform(self, image):
        # Compute similarity transform of image and show it
        image_similarity = cv2.warpAffine(image, self.M_sim, image.shape[:2])
        cv2.imshow("Image Similarity", image_similarity)
        cv2.waitKey(0)

    def similarityError(self):
        # Compute similarity transformation over pts1 and compute error wrt pts 2
        M_pts = np.append(self.M_sim, np.array([[0, 0, 1]]), axis=0)
        pts = np.append(self.pts1.transpose(), np.array([[1, 1, 1]]), axis=0)
        pts_transform = np.matmul(M_pts, pts)
        pts_transform = pts_transform[:-1,:].transpose()

        error = np.linalg.norm(pts_transform - self.pts2, axis=1)

        # print(pts_transform)
        # print(self.pts2)
        print('Error = ', error)


def capturePoints(event, x, y, flags, params):
    # Capture mouse events
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        # print(refPt)

if __name__ == '__main__':
    # Paths of image 1 and 2
    path = os.path.dirname(__file__)
    path_file1 = os.path.join(path, '../imgs/lena.png')
    path_file2 = os.path.join(path, '../imgs/lena_warped.png')

    # Read images and guarantee width = height
    image1 = cv2.imread(path_file1)
    image1 = cv2.resize(image1, (512, 512), interpolation=cv2.INTER_CUBIC)
    image2 = cv2.imread(path_file2)
    image2 = cv2.resize(image2, (512, 512), interpolation=cv2.INTER_CUBIC)

    # Show image 1 and capture points indicated with the mouse: expects 3 points
    refPt = []
    cv2.imshow('Image1', image1)
    cv2.setMouseCallback('Image1', capturePoints)
    cv2.waitKey(0)
    puntos1 = refPt
    pts1 = np.float32(puntos1)

    # Show image 2 and capture points indicated with the mouse: expects 3 points
    refPt = []
    cv2.imshow('Image2', image2)
    cv2.setMouseCallback('Image2', capturePoints)
    cv2.waitKey(0)
    puntos2 = refPt
    pts2 = np.float32(puntos2)

    # Compute affine transform matrix, transform image 1 and show result
    affine = geomTransform(pts1, pts2)
    affine.affineTransform(image1)

    # Compute similarity transformation and show result
    affine.estSimilarity()
    affine.similarityTransform(image1)

    # Compute difference betweeen the transformation of image 1 and image 2
    affine.similarityError()


