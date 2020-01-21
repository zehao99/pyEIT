from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.eit.efem import Forward

""" 0. construct mesh """
mesh_obj, el_pos = mesh.create(16, h0=0.1)
# extract node, element, alpha
points = mesh_obj['node']
tri = mesh_obj['element']
x, y = points[:, 0], points[:, 1]

""" 1. problem setup """
fwd = Forward(mesh_obj,[tri[0],tri[1]])
fwd.elem_perm = 10 * fwd.elem_perm
#fwd.change_capacity([100,101,102,103,104,105],[100,100,100,100,100,100])
_ , elem_u = fwd.calculation()

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.tripcolor(x, y, tri, np.abs(elem_u), shading='flat')
fig.colorbar(im)
ax.set_aspect('equal')

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.tripcolor(x, y, tri, np.real(fwd.elem_capacity), shading='flat')
fig.colorbar(im)
ax.set_aspect('equal')

plt.show()