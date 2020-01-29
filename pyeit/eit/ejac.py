from __future__ import division, absolute_import, print_function

import numpy as np 
import cupy as cp
from pyeit.eit.efem import EFEM
import math
import time
import progressbar
import csv

import matplotlib.pyplot as plt
import pyeit.mesh as mesh

#Depend on efem.py

class EJAC(object):
    """
    calculate Jaccobian matrix for the problem

    Dimension: patterns_num * element numbers
    """
    def __init__(self, mesh, electrode_num = 4, frequency = 20000.0):
        #Create FEM class
        self.fwd_FEM = EFEM(mesh, electrode_num, frequency)
        self.electrode_num = electrode_num 
        #Set overall permitivity 
        perm = 10
        self.fwd_FEM.elem_perm = perm * self.fwd_FEM.elem_perm
        #Set electrode meshes(Circle)
        radius = 0.1
        for i in range(electrode_num):
            theta = i * math.pi/electrode_num
            center_x = math.cos(theta)
            center_y = math.sin(theta)
            center = [center_x,center_y]
            self.fwd_FEM.calc_electrode_elements(i,center, radius)
        #Initialize Matrix
        self.pattern_num = electrode_num * (electrode_num - 1)
        self.elem_num = self.fwd_FEM.elem_num
        self.electrode_original_potential = np.zeros((self.pattern_num))
        self.JAC_matrix = np.zeros((self.pattern_num,self.elem_num))
        self.calc_origin_potential()
    
    def calc_origin_potential(self):
        """
        calculate origin potential vector
        """
        for i in range(self.electrode_num):
            count = 0
            self.fwd_FEM.calculation(i)
            for m in range(self.electrode_num):
                if m != i:
                    self.electrode_original_potential[i * (self.electrode_num - 1) + count] = np.abs(self.fwd_FEM.electrode_potential[m]) 
                    count += 1

    def JAC_calculation(self):
        
        #capacity change in JAC
        capacity_change = 0.00001
        #Changed matrix
        for i in range(self.electrode_num):
            print("iteration: " + str(i))
            for j in progressbar.progressbar(range(self.elem_num)):
                time.sleep(0.001)
                self.fwd_FEM.elem_capacity[j] = capacity_change
                self.fwd_FEM.calculation(i)
                count = 0
                for m in range(self.electrode_num):
                    if m != i:
                        self.JAC_matrix[i * (self.electrode_num - 1) + count][j] = np.abs(self.fwd_FEM.electrode_potential[m])
                        count += 1
                self.fwd_FEM.elem_capacity[j] = 0
        #Minus and broadcast original value calculate differiential value
        self.JAC_matrix = (self.JAC_matrix - np.reshape(self.electrode_original_potential, (self.pattern_num,1))) / capacity_change
        return self.JAC_matrix

    def eit_solve(self, detect_potential):
        """
        detect_potential: electrode_num * (electrode_num - 1) elements NDArray vector
        """
        J = np.copy(self.JAC_matrix)
        Q = np.eye(self.elem_num)
        #regularisation parameter
        lmbda = 1
        delta_V = detect_potential - self.electrode_original_potential
        capacity_predict = np.dot(np.dot(np.linalg.inv(np.dot(J.T, J) + lmbda ** 2 * Q ), J.T), delta_V)
        return capacity_predict

    def save_JAC_2file(self, filename):
        """
        Save jaccobian matrix to file
        parameter: filename string
        """
        with open('jac_cache_'+ filename +'.csv', "w", newline= '') as csvfile:
            writer = csv.writer(csvfile, delimiter = ' ')
            for row in self.JAC_matrix:
                writer.writerow(row)
    
    def read_JAC_2file(self, filename):
        """
        Read jaccobian matrix from file
        parameter: filename string
        """
        with open('jac_cache_'+ filename +'.csv', newline= '') as csvfile:
            reader = csv.reader(csvfile, delimiter = ' ')
            for i, line in enumerate(reader):
                self.JAC_matrix[i] = line

""" 0. construct mesh """
mesh_obj, el_pos = mesh.create(16, h0=0.1)
# extract node, element, alpha
points = mesh_obj['node']
tri = mesh_obj['element']
x, y = points[:, 0], points[:, 1]

""" 1. problem setup """
fwd = EJAC(mesh_obj)
JAC_matrix = fwd.JAC_calculation()
fwd.save_JAC_2file("fwd00000")

""" 2. simulation setup """
experiment = EJAC(mesh_obj)
experiment.fwd_FEM.change_capacity_geometry([0,0.1], 0.2, 0.001, "circle")

elem_capacity = experiment.fwd_FEM.elem_capacity
experiment.calc_origin_potential()
electrode_potential = experiment.electrode_original_potential
""" 3. solve eit problem """
c_predict = fwd.eit_solve(electrode_potential)

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.tripcolor(x, y, tri, np.abs(c_predict), shading='flat')
fig.colorbar(im)
ax.set_aspect('equal')

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.tripcolor(x, y, tri, np.real(elem_capacity), shading='flat')
fig.colorbar(im)
ax.set_aspect('equal')

plt.show()