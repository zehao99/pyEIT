from __future__ import division, absolute_import, print_function
import math
import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.eit.efem import EFEM

""" 0. construct mesh """
mesh_obj, el_pos = mesh.create(16, h0=0.1)
# extract node, element, alpha
points = mesh_obj['node']
tri = mesh_obj['element']
x, y = points[:, 0], points[:, 1]

""" 1. problem setup """
electrode_num = 16
fwd = EFEM(mesh_obj,electrode_num)
radius = 0.1
for i in range(electrode_num):
    theta =2* i * math.pi/electrode_num
    center_x = math.cos(theta)
    center_y = math.sin(theta)
    center = [center_x,center_y]
    fwd.calc_electrode_elements(i,center, radius)
fwd.elem_perm = 10 * fwd.elem_perm
fwd.change_capacity_geometry([0,0.1], 0.2, 0.01, "circle")
potentials = np.zeros((electrode_num,fwd.elem_num), dtype = np.complex128)
for i in range(electrode_num):
    _ , elem_u, elem_potential = fwd.calculation(i)
    potentials[i] = elem_u

print(elem_potential)

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.tripcolor(x, y, tri, np.abs(elem_u), shading='flat')
fig.colorbar(im)
ax.set_aspect('equal')

fig, ax = plt.subplots(4,4,figsize=(24, 16))
levels = np.arange(0,1,0.1)
vmin = np.min(np.abs(potentials))
for i in range(electrode_num):
    fig = plt.subplot(4,4,i+1)
    im = fig.tripcolor(x, y, tri, np.abs(potentials[i]), shading='flat', vmax = 1, vmin = vmin)
    plt.colorbar(im)
    fig.set_aspect('equal')

plt.show()