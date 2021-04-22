"""
Analyse INPA distortion

Jose Rueda: jrrueda@us.es

Note. I tryed with some fancy method to automatically detect the grids, but at
the end it was giving problems and this is something we will do at most once
per campaing, so I changed to a ginput method
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------
file1 = '/afs/ipp/home/r/ruejo/INPA_calibration/data/cuadricula.jpeg'  # real
file2 = '/afs/ipp/home/r/ruejo/INPA_calibration/data/distorted.tif'  # distort
file_zeemax = '/afs/ipp/home/r/ruejo/ScintSuite/Data/OpticalModelling/' +\
    'INPA1AUG_distortion_Zeemax.txt'
# -----------------------------------------------------------------------------
# --- Load images
# -----------------------------------------------------------------------------
img1 = cv2.imread(file1)
img2 = cv2.imread(file2)

# -----------------------------------------------------------------------------
# --- Select grids
# -----------------------------------------------------------------------------
# Note, you should select the grid points in the same order in each grid!!!

plt.imshow(img1)
points1 = plt.ginput(-1, timeout=0)

plt.close('all')
plt.imshow(img2)
points2 = plt.ginput(-1, timeout=0)

# -----------------------------------------------------------------------------
# --- Select the centers and scales
# -----------------------------------------------------------------------------
# Select the vertex closer to the optical axis, this must be done manually
center_v = 36
p_center1 = np.array(points1[center_v])
p_center2 = np.array(points2[center_v])
# Select the vertex closer to this center one, which would be used to find the
# proper scale in each vertex
near_points = [25, 35, 37, 49]
# Calculate the distances to the center in both images:
npoints = len(near_points)
d1 = []
d2 = []
for i in near_points:
    d1.append(np.sqrt(np.sum((p_center1 - np.array(points1[i]))**2)))
    d2.append(np.sqrt(np.sum((p_center2 - np.array(points2[i]))**2)))
# Calculate the distance to the center for all points
npoints_complete = len(points1)
dc1 = []
dc2 = []
for i in range(npoints_complete):
    dc1.append(np.sqrt(np.sum((p_center1 - np.array(points1[i]))**2)))
    dc2.append(np.sqrt(np.sum((p_center2 - np.array(points2[i]))**2)))

# -----------------------------------------------------------------------------
# --- Scale boths distances arrays to the same scale
# -----------------------------------------------------------------------------
dc1_scaled = np.array(dc1) / np.array(d1).mean()
dc2_scaled = np.array(dc2) / np.array(d2).mean()
# Note the original grid was spaced 1 cm

fig, ax = plt.subplots()
ax.plot(dc1_scaled, dc1_scaled, '.k', label='Ideal')
ax.plot(dc1_scaled, dc2_scaled, '.r', label='Distorted')

# -----------------------------------------------------------------------------
# --- Calculate the distortion
# -----------------------------------------------------------------------------
distortion = 100 * (dc2_scaled - dc1_scaled) / dc1_scaled

fig2, ax2 = plt.subplots()
ax2.plot(dc1_scaled, distortion, '.k', label='Measured')

# --- Load the original
data_zeemax = np.loadtxt(file_zeemax, skiprows=1)
ax2.plot(data_zeemax[:, 0], data_zeemax[:, 1], '.r', label='Zeemax')
ax2.set_xlabel('Distance in the Scintillator')
ax2.set_ylabel('Distortion [%]')
# -----------------------------------------------------------------------------
# --- Plot the grids
# -----------------------------------------------------------------------------
x1 = np.zeros(npoints_complete)
y1 = np.zeros(npoints_complete)

x2 = np.zeros(npoints_complete)
y2 = np.zeros(npoints_complete)

for i in range(npoints_complete):
    x1[i] = points1[i][0]
    y1[i] = points1[i][1]
    x2[i] = points2[i][0]
    y2[i] = points2[i][1]
fig3, ax3 = plt.subplots()
ax3.imshow(img1)
ax3.plot(x1, y1, '.b')
ax3.plot(x1[center_v], y1[center_v], '.r')

fig4, ax4 = plt.subplots()
ax4.imshow(img2)
ax4.plot(x2, y2, '.b')
ax4.plot(x2[center_v], y2[center_v], '.r')
# -----------------------------------------------------------------------------
# the magnification factor is just the inverse of d2.mean() in px/cm
