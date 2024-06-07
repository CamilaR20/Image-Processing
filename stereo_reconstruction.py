#!/usr/bin/env python

"""
Simple example of stereo image matching and point cloud generation.
Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
"""

import numpy as np
import cv2 as cv
import sys
import os

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def main():
    print('loading images...')
    path = os.path.join(os.path.dirname(__file__), 'imgs/teddy')
    path_file1 = os.path.join(path, 'im1.png')
    path_file2 = os.path.join(path, 'im2.png')
    imgL = cv.pyrDown(cv.imread(path_file1))
    imgR = cv.pyrDown(cv.imread(path_file2))

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 1
    num_disp = 128
    stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                  numDisparities=num_disp,
                                  blockSize=15,
                                  P1=2,
                                  P2=5,
                                  disp12MaxDiff=1,
                                  uniquenessRatio=20,
                                  speckleWindowSize=100,
                                  speckleRange=4
                                  )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print('generating 3d point cloud...', )

    rows, cols = imgL.shape[:2]
    enum_rows = np.linspace(0, rows - 1, rows)
    enum_cols = np.linspace(0, cols - 1, cols)
    col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)

    pixel_x = col_iter.flatten()
    pixel_y = row_iter.flatten()
    pixel_z = np.ones_like(pixel_x)
    pixels = [pixel_x, pixel_y, pixel_z]

    fx = 1000
    fy = 1000
    cx = cols // 2
    cy = rows // 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    baseline = 100
    f = 1000
    Z = (baseline * f) / (disp.flatten() + 0.0001)
    points = Z * np.matmul(np.linalg.inv(K), pixels)
    points = points.T

    colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    points = points.reshape((rows, cols, 3))
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_points = out_points[::2]
    out_colors = out_colors[::2]
    out_fn = 'out.ply'
    write_ply(out_fn, out_points, out_colors)
    print('%s saved' % out_fn)

    cv.imshow('left', imgL)
    cv.imshow('disparity', (disp - min_disp) / num_disp)
    cv.waitKey()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
