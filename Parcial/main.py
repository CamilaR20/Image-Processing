import sys
import os
from bandera import *


if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)

    bandera = Bandera(image)    # Objeto de clase Bandera
    n_colores = bandera.colores()
    print("Número de colores de la bandera: ", n_colores)
    porcentajes = bandera.porcentaje()
    porcentajes = [round(n, 3) for n in porcentajes]
    print("Porcentaje de cada color de la bandera: ", porcentajes)
    orient = bandera.orientacion()
    print("Orientacion de la bandera: ", orient)



