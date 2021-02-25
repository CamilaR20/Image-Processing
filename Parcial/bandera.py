import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from hough import hough


class Bandera:
    def __init__(self, im):
        # Recibe una imagen y la guarda en el objeto
        self.im = im
        self.width, self.height, self.ch = im.shape
        assert self.ch == 3

    def colores(self):
        # Retorna el numero de colores de la imagen de entrada
        # Cambia imágen a representación RGB y a flotante
        im_rgb = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
        im_rgb = np.array(self.im, dtype=np.float64) / 255

        # Guarda la imagen convertida a un arreglo de 2D
        self.im_array = np.reshape(im_rgb, (self.width * self.height, self.ch))
        # Función para encontrar el número de colores
        self.n_color, self.centers, self.labels = color_seg(self.im_array)

        return self.n_color

    def porcentaje(self):
        porcentajes = []
        for i in range(self.centers.shape[0]):
            p_color = np.count_nonzero(self.labels == i) / self.labels.shape[0]
            porcentajes.append(p_color)
        return porcentajes

    def orientacion(self):
        # Retorna la orientacion de las lineas de la bandera
        high_thresh = 300
        bw_edges = cv2.Canny(self.im, high_thresh * 0.3, high_thresh, L2gradient=True)
        hough_obj = hough(bw_edges)
        accumulator = hough_obj.standard_HT()

        acc_thresh = 50
        N_peaks = 4
        nhood = [25, 9]
        peaks = hough_obj.find_peaks(accumulator, nhood, acc_thresh, N_peaks)

        vert = False
        horiz = False
        for i in range(len(peaks)):
            theta_ = hough_obj.theta[peaks[i][1]]
            theta_ = theta_ - 180
            theta_ = int(theta_)

            if (abs(theta_) == 0) or (abs(theta_) == 180):
                vert = True
                orient_str = "Vertical"
            elif(abs(theta_) == 90):
                horiz = True
                orient_str = "Horizontal"

        if (vert and horiz):
            orient_str = "Mixta"

        return orient_str


def color_seg(im_array):
    # Encuentra el número de clusters de 1 a 4 que tenga la mínima distancia intracluster
    im_array_sample = shuffle(im_array, random_state=0)[:10000]
    num_clusters = 4
    distancias = np.zeros((num_clusters, 1))
    for n_color in range(1, num_clusters + 1):
        # Para n_color calcular clustering
        model = KMeans(n_clusters=n_color, random_state=0).fit(im_array_sample)
        labels = model.predict(im_array)
        centers = model.cluster_centers_
        distancias[n_color - 1] = model.inertia_

    # Solo toma la primera ocurrencia del mínimo
    n_color = int(distancias.argmin()) + 1
    # model = KMeans(n_clusters=n_color, random_state=0).fit(im_array_sample)
    # labels = model.predict(im_array)
    # centers = model.cluster_centers_

    return n_color, centers, labels

if __name__ == '__main__':
    pass

