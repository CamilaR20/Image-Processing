import cv2
import numpy as np


class imageShape:
    def __init__(self, width=512, height=512):
        # Receives parameters of the image and saves them
        self.width = width
        self.height = height
        self.color = (255, 255, 0)

    def generateShape(self):
        # Generates image with dimensions specified in the constructor
        self.shape = np.zeros((self.height, self.width, 3), dtype='uint8')
        im_center = np.array((self.width / 2 - 1, self.height / 2 - 1), dtype='int')

        # According to a random number a shape is chosen to be drawn in the image
        shapes_list = ['triangle', 'square', 'rectangle', 'circle']
        self.shape_name = shapes_list[np.random.randint(4)]
        # self.shape_name = shapes_list[2]

        if self.shape_name == 'triangle':
            # Computes side and vertices of the triangle to draw it
            side = min(self.width, self.height) / 2
            R = side / np.sqrt(3)
            r = R / 2
            pts = np.array(((im_center[0], im_center[1] - R),
                            (im_center[0] - side / 2, im_center[1] + r),
                            (im_center[0] + side / 2, im_center[1] + r)))
            pts = pts.astype(int)
            self.shape = cv2.fillPoly(self.shape, [pts], self.color)

        elif self.shape_name == 'square':
            # Computes sides and vertices assuming a (0,0) center
            side = min(self.width, self.height) / 2
            pts = np.array(((-side / 2, side / 2), (side / 2, side / 2),
                            (side / 2, -side / 2), (-side / 2, -side / 2)))

            # Rotates figure according to rotation matrix
            c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
            R = np.array(((c, -s), (s, c)))
            pts = np.dot(pts, R.T)
            pts = pts.astype(int)

            # Translates figure to the center of the image and draws it
            pts = pts + im_center
            self.shape = cv2.fillPoly(self.shape, [pts], self.color)

        elif self.shape_name == 'rectangle':
            # Computes sides of the rectangle
            horizontal = self.width / 2
            vertical = self.height / 2

            # Computes starting and endpoint to draw rectangle
            start_point = (int(im_center[0] - horizontal / 2), int(im_center[1] - vertical / 2))
            end_point = (int(start_point[0] + horizontal), int(start_point[1] + vertical))
            self.shape = cv2.rectangle(self.shape, start_point, end_point, self.color, -1)

        elif self.shape_name == 'circle':
            # Computes radius and draws circle
            radius = int(min(self.width, self.height) / 4)
            self.shape = cv2.circle(self.shape, (im_center[0], im_center[1]), radius, self.color, -1)

    def showShape(self):
        # Shows image for 5s, if there is no shape it shows a black image
        if hasattr(self, 'shape'):
            cv2.imshow(self.shape_name, self.shape)
            cv2.waitKey(5000)
        else:
            black_img = np.zeros((self.height, self.width), dtype='uint8')
            cv2.imshow("Image", black_img)
            cv2.waitKey(5000)

    def getShape(self):
        # Returns the image saved in the object and the shape name within it
        return self.shape, self.shape_name

    def whatShape(self, image):
        # Converts the image to grayscale and uses Otsu to obtain shape
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, im_bw = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Calcula contorno de la figura
        contour, hierarchy = cv2.findContours(im_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Aproxima contorno a un polígono
        epsilon = 0.01 * cv2.arcLength(contour[0], True)
        approx = cv2.approxPolyDP(contour[0], epsilon, True)
        vertices = approx.shape[0]

        # According to the number of vertices it classifies the figure
        if vertices == 3:
            return "triangle"
        elif vertices == 4:
            # Use aspect ration of rectangle (min area rectangle) that encloses the contour, to differentiate between square and rectangle
            rect = cv2.minAreaRect(contour[0])
            aspect_ratio = float(rect[1][0] / rect[1][1])
            if 0.99 <= aspect_ratio <= 1.01:
                return "square"
            else:
                return "rectangle"
        else:
            return "circle"


if __name__ == '__main__':
    pass
