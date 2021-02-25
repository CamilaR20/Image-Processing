import cv2
import numpy as np
import sys
import os

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time


def recreate_image(centers, labels, rows, cols):
    d = centers.shape[1]
    image_clusters = np.zeros((rows, cols, d))
    label_idx = 0
    for i in range(rows):
        for j in range(cols):
            image_clusters[i][j] = centers[labels[label_idx]]
            label_idx += 1
    return image_clusters

# def recreate_image(centers, labels, rows, cols):
#     ch = centers.shape[1]
#     image_clusters = np.zeros((rows * cols, ch))
#     for i in range(centers.shape[0]):
#         image_clusters[labels == i] = centers[i]
#     image_clusters = np.reshape(image_clusters, (rows, cols, ch))
#     return image_clusters


if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    n_colors = 5
    method = ['kmeans', 'gmm']
    select = 0

    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow behaves works well on float data (need to
    # be in the range [0-1])
    image = np.array(image, dtype=np.float64) / 255

    # Load Image and transform to a 2D numpy array.
    rows, cols, ch = image.shape
    assert ch == 3
    image_array = np.reshape(image, (rows * cols, ch))

    print("Fitting model on a small sub-sample of the data")
    t0 = time()
    image_array_sample = shuffle(image_array, random_state=0)[:10000]
    if method[select] == 'gmm':
        model = GMM(n_components=n_colors).fit(image_array_sample)
    else:
        model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    print("done in %0.3fs." % (time() - t0))

    # Get labels for all points
    print("Predicting color indices on the full image (GMM)")
    t0 = time()
    if method[select] == 'gmm':
        labels = model.predict(image_array)
        centers = model.means_
    else:
        labels = model.predict(image_array)
        centers = model.cluster_centers_
    print("done in %0.3fs." % (time() - t0))

    # Display all results, alongside original image
    plt.figure(1)
    plt.clf()
    plt.axis('off')
    plt.title('Original image')
    plt.imshow(image)

    plt.figure(2)
    plt.clf()
    plt.axis('off')
    plt.title('Quantized image ({} colors, method={})'.format(n_colors, method[select]))
    plt.imshow(recreate_image(centers, labels, rows, cols))

    plt.show()



