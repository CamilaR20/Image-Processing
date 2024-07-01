import cv2
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

class colorSeg:
    def __init__(self, path, method):
        # Saves image and clustering method
        self.image = cv2.imread(path)
        self.method = method

        # Changes the image to RGB and float
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = np.array(self.image, dtype=np.float64) / 255

        # Saves the size of the image and verifies it has 3 channels
        self.rows, self.cols, ch = self.image.shape
        assert ch == 3
        # Saves the image into a 2D array
        self.image_array = np.reshape(self.image, (self.rows * self.cols, ch))

    def clustering(self, n_color):
        # Compute color segmentation according to method specified in the constructor
        self.n_color = n_color
        image_array_sample = shuffle(self.image_array, random_state=0)[:10000]
        if self.method == 'gmm':
            model = GMM(n_components=n_color).fit(image_array_sample)
            self.labels = model.predict(self.image_array)
            self.centers = model.means_
        else:
            model = KMeans(n_clusters=n_color, random_state=0).fit(image_array_sample)
            self.labels = model.predict(self.image_array)
            self.centers = model.cluster_centers_

    def calc_dist(self):
        # Compute sum of intra-cluster distances for a value of n_color
        intracluster = 0
        for label in range(self.centers.shape[0]):
            vector_label = self.image_array[self.labels == label]
            resta = vector_label - self.centers[label]
            magnitude = np.linalg.norm(resta, axis=1)
            distancia = np.sum(magnitude)
            intracluster = intracluster + distancia
        return intracluster


    def recreate_image(self, fig):
        # Show segmented image for a value of n_color
        d = self.centers.shape[1]
        image_clusters = np.zeros((self.rows, self.cols, d))
        label_idx = 0
        for i in range(self.rows):
            for j in range(self.cols):
                image_clusters[i][j] = self.centers[self.labels[label_idx]]
                label_idx += 1

        plt.figure(fig)
        plt.clf()
        plt.axis('off')
        plt.title('Quantized image ({} colors, method={})'.format(self.n_color, self.method))
        plt.imshow(image_clusters)
        plt.show()


if __name__ == '__main__':
    path_file = os.path.join(os.path.dirname(__file__), 'imgs/flag.png')
    # Clustering method kmeans or gmm
    method = 'kmeans'
    imagen = colorSeg(path_file, method)

    # Compute intra-cluster distance for 1 to 10 clusters
    num_clusters = 10
    distancias = np.zeros((num_clusters, 1))
    for n_color in range(1, num_clusters + 1):
        imagen.clustering(n_color)     # Compute clustering for n_color
        imagen.recreate_image(n_color) # Show image segmented for n_color
        distancias[n_color - 1] = imagen.calc_dist() # Compute intra-cluster distance for n_color clusters

    # Plot sum of intra-cluster distances vs n_color
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, num_clusters+1), distancias, marker='o')
    plt.title('Sum of intra-cluster distances vs number of colors, ', method)
    plt.xlabel('Number of colors')
    plt.ylabel('Sum of intra-cluster distances')
    plt.xticks(np.arange(0, num_clusters+1, 1))
    plt.show()


