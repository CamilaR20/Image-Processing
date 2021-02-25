import cv2
import numpy as np


class imageShape:
    def __init__(self, width=512, height=512):
        # Recibe parámetros de la imagen y las guarda en el objeto
        self.width = width
        self.height = height
        self.color = (255, 255, 0)

    def generateShape(self):
        # Genera y almacena una imagen en el objeto con las dimensiones indicadas al constructor
        self.shape = np.zeros((self.height, self.width, 3), dtype='uint8')
        im_center = np.array((self.width / 2 - 1, self.height / 2 - 1), dtype='int')

        # De acuerdo con un número aleatorio decide que figura se dibujará en la imagen
        shapes_list = ['triangle', 'square', 'rectangle', 'circle']
        self.shape_name = shapes_list[np.random.randint(4)]
        # self.shape_name = shapes_list[2]

        if self.shape_name == 'triangle':
            # Calcula lado del triángulo para calcular vértices y dibujar triángulo
            side = min(self.width, self.height) / 2
            R = side / np.sqrt(3)
            r = R / 2
            pts = np.array(((im_center[0], im_center[1] - R),
                            (im_center[0] - side / 2, im_center[1] + r),
                            (im_center[0] + side / 2, im_center[1] + r)))
            pts = pts.astype(np.int)
            self.shape = cv2.fillPoly(self.shape, [pts], self.color)

        elif self.shape_name == 'square':
            # Calcula parámetro (lado) y vértices suponiendo centro (0,0)
            side = min(self.width, self.height) / 2
            pts = np.array(((-side / 2, side / 2), (side / 2, side / 2),
                            (side / 2, -side / 2), (-side / 2, -side / 2)))

            # Rota la figura con matriz de rotación
            c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
            R = np.array(((c, -s), (s, c)))
            pts = np.dot(pts, R.T)
            pts = pts.astype(np.int)

            # Lleva figura al centro de la imagen y dibuja
            pts = pts + im_center
            self.shape = cv2.fillPoly(self.shape, [pts], self.color)

        elif self.shape_name == 'rectangle':
            # Calcula parámetros del rectángulo (lados)
            horizontal = self.width / 2
            vertical = self.height / 2

            # Calcula punto de inicio y punto final para dibujar el rectángulo
            start_point = (int(im_center[0] - horizontal / 2), int(im_center[1] - vertical / 2))
            end_point = (int(start_point[0] + horizontal), int(start_point[1] + vertical))
            self.shape = cv2.rectangle(self.shape, start_point, end_point, self.color, -1)

        elif self.shape_name == 'circle':
            # Calcula radio y dibuja el círculo
            radius = int(min(self.width, self.height) / 4)
            self.shape = cv2.circle(self.shape, (im_center[0], im_center[1]), radius, self.color, -1)

    def showShape(self):
        # Muestra la imagen almacenada en el objeto por 5s,
        # si no hay figura muestra imagen en negro
        if hasattr(self, 'shape'):
            cv2.imshow(self.shape_name, self.shape)
            cv2.waitKey(5000)
        else:
            black_img = np.zeros((self.height, self.width), dtype='uint8')
            cv2.imshow("Image", black_img)
            cv2.waitKey(5000)

    def getShape(self):
        # Devuelve imagen guardada en el objeto y el nombre de la figura
        return self.shape, self.shape_name

    def whatShape(self, image):
        # Pasa imagen a escala de grises y umbraliza con método de Otsu para obtener figura
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, im_bw = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Calcula contorno de la figura
        contour, hierarchy = cv2.findContours(im_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Aproxima contorno a un polígono
        epsilon = 0.01 * cv2.arcLength(contour[0], True)
        approx = cv2.approxPolyDP(contour[0], epsilon, True)
        vertices = approx.shape[0]

        # Según el número de vértices del polígono aproximado clasifica la figura
        if vertices == 3:
            return "triangle"
        elif vertices == 4:
            # Usa el aspect ratio del rectángulo (min area rectangle) que encierra el contorno,
            # para diferenciar entre un cuadrado y un rectángulo
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
