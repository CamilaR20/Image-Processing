import sys
import os
import cv2
from flag import Flag


if __name__ == '__main__':
    path_file = os.path.join(os.path.dirname(__file__), '../imgs/flag1.png')
    print(path_file)
    image = cv2.imread(path_file)

    flag = Flag(image)
    n_colors = flag.colores()
    print("Number of colors of the flag: ", n_colors)
    percentages = flag.percentage()
    percentages = [round(n, 3) for n in percentages]
    print("Percentages of each color of the flag: ", percentages)
    orient = flag.orientation()
    print("Flag orientation: ", orient)



