from __future__ import division, absolute_import, print_function
from numba import jit


import numpy as np 
import cupy as cp
import time
import math

import matplotlib.pyplot as plt
import pyeit.mesh as mesh


class EFEM(object):
    """
    FEM solving for capacity included forward problem
    """

    def __init__(self, mesh, electrode_nums, freqency = 20000.0):
        """
        Parameters:
        mesh: dict with "node", "element" "perm" "capacity"
            "node": N_node * 2 NDArray 
                    contains coordinate information of each node
            "element": N_elem * 3 NDArray
                    contains the node number contained in each element
            "perm": N_elem NDArray
                    contains permitivity on each element
            "capacity": N_elem NDArray
                    contains capacity density on each element
    
        electrode_nums:  int electrode numbers
    
        frequency : float Input frequency
        """
        #check data structure
        self.nodes = mesh['node']
        self.node_num = np.shape(self.nodes)[0]
        if np.shape(self.nodes)[1] != 2:
            raise Exception("Node coordinates incorrect")
        self.node_num_bound = 0
        self.node_num_f = 0
        self.elem =  mesh['element']
        if np.shape(self.elem)[1] != 3:
            raise Exception("Elements incorrect")
        self.elem_num = np.shape(self.elem)[0]
        self.elem_perm = mesh['perm']
        self.elem_capacity = np.zeros(np.shape(self.elem_perm))
        self.elem_param = np.zeros((np.shape(self.elem)[0],9)) # area, b1, b2, b3, c1, c2, c3, x_average, y_average
        self.electrode_num = electrode_nums
        self.electrode_mesh = dict()
        for i in range(electrode_nums):
            self.electrode_mesh[i] = list()
        self.freq = freqency
        self.K_sparse = np.zeros((self.node_num,self.node_num),dtype = np.complex128)
        self.K_node_num_list = [x for x in range(self.node_num)]# Node number mapping list when calculating
        self.node_potential = np.zeros((self.node_num), dtype = np.complex128)
        self.element_potential = np.zeros((self.elem_num),dtype = np.complex128)
        self.electrode_potential = np.zeros((self.electrode_num), dtype = np.complex128)
        self.initialize()
        
    def calculation(self,electrode_input):
        """
        Parameters: electrode_input input position
        """
        self.construct_sparse_matrix()
        self.set_boundary_condition(electrode_input)
        #split time into frames
        frames = 8
        temp_node_potential = np.zeros((self.node_num), dtype = np.complex128)        
        for i in range(frames):
            theta = np.float(2*np.pi/frames * i)
            temp_node_potential += np.abs(self.calculate_FEM(theta))
        temp_node_potential /= frames
        self.node_potential = temp_node_potential
        self.sync_back_potential()
        self.calculate_element_potential()
        self.calc_electrode_potential()
        return self.node_potential, self.element_potential, self.electrode_potential

    def initialize(self):
        """
        Update parameters for each element
        Parameters used for calculating sparse matrix
        a1 = x2 * y3 - x3 * y2
        b1 = y2 - y3
        c1 = x3 - x2
        area = (b1 * c2 - b2 * c1) / 2
        """
        x = [.0,.0,.0]
        #a = [.0,.0,.0]
        b = [.0,.0,.0]
        c = [.0,.0,.0]
        y = [.0,.0,.0]
        count = 0
        for element in self.elem:
            for i in range(3):
                x[i] = self.nodes[element[i],0]
                y[i] = self.nodes[element[i],1]
            for i in range(3):
                #a[i] = x[(1+i)%3] * y[(2+i)%3] - x[(2+i)%3] * y[(1+i)%3]
                b[i] = y[(1+i)%3] - y[(2+i)%3]
                c[i] = x[(2+i)%3] - x[(1+i)%3]
            area = (b[0] * c[1] - b[1] * c[0])/2
            x_average = np.mean(x)
            y_average = np.mean(y)
            self.elem_param[count] = [area, b[0], b[1], b[2], c[0], c[1], c[2], x_average, y_average]
            count += 1
    
    def construct_sparse_matrix(self):
        """
        construct the original sparse matrix 
        """
        index = 0
        K_ij = 0 + 0j
        patern = [[0,0],[1,1],[2,2],[0,1],[1,2],[2,0],[1,0],[2,1],[0,2]] # go through every combination of k1 and k2
        for element in self.elem:
            param = self.elem_param[index]
            for i,j in patern:
                if i != j :
                     # stiffness k_ij = sigma * (bk1*bk2 + ck1*ck2)/(4 * area) - j * w * capacity * (bk1 * ck2 - bk2 * ck1) /24
                    K_ij = self.elem_perm[index] * (param[1+i] * param[1+j] + param[4+i] * param[4+j]) / (4 * param[0]) -  (self.freq * self.elem_capacity[index] * param[0] /12) * 1j
                else:
                    K_ij = self.elem_perm[index] * (param[1+i] * param[1+j] + param[4+i] * param[4+j]) / (4 * param[0]) - (self.freq * self.elem_capacity[index] * param[0] /6) * 1j
                self.K_sparse[element[i]][element[j]] += K_ij
                self.K_sparse[element[j]][element[i]] += K_ij
                
                """debug part
                if self.elem_capacity[index] != 0:
                    print(self.elem_capacity[index])
                    print((self.freq * self.elem_capacity[index] * param[0] /6) * 1j)
                if np.imag( K_ij ) != 0:
                    print(K_ij)
                """
            index += 1

    def set_boundary_condition(self, electrode_input):
        """
        Update boundary condition according to electrode mesh

        The boundary is at the input electrode whose potential is all Ae ^ iPhi

        And swap the index of matrix put the boundary elements at the bottom of the sparse matrix
        """
        node_list = [] # reshape all nodes to 1D
        electrode_list = list(self.electrode_mesh.values())
        for element in electrode_list[electrode_input]:
            node_list.append(self.elem[element][0])
            node_list.append(self.elem[element][1])
            node_list.append(self.elem[element][2])
        node_list = np.array(node_list)
        node_list = list(np.unique(node_list)) # get rid of repeat numbers 
        index = self.node_num
        self.node_num_bound = len(node_list)
        self.node_num_f = self.node_num - self.node_num_bound
        for list_num in node_list:
            index = index - 1
            self.swap(list_num, index)

    def change_capacity_elementwise(self, element_list, capacity_list):
        """
        Change capacity in certain area according to ELEMENT NUMBER
        """
        if len(element_list) == len(capacity_list):
            for i, ele_num in enumerate(element_list):
                if ele_num > self.elem_num:
                    raise Exception("Element number exceeds limit")
                self.elem_capacity[ele_num] = capacity_list[i]
        else:
            raise Exception('The length of element doesn\'t match the length of capacity')

    def change_capacity_geometry(self, center, radius, value, shape):
        """
        Parameters: shape: "circle", "square"

        Change capacity in certain area according to GEOMETRY
        """
        if shape == "square":
            center_x, center_y = center
            count = 0
            for i, x in enumerate(self.elem_param[:, 7]):
                if x <= (center_x + radius) and x >= (center_x - radius) and self.elem_param[i][8] <= (center_y + radius) and self.elem_param[i][8] >= (center_y - radius):
                    self.elem_capacity[i] = value
                    count += 1                    
            if count == 0:
                raise Exception("No element is selected, please check the input")
        elif shape == "circle":
            center_x, center_y = center
            count = 0
            for i, x in enumerate(self.elem_param[:, 7]):
                if np.sqrt((center_x - x)**2+(center_y - self.elem_param[i][8])**2) <= radius:
                    self.elem_capacity[i] = value
                    count += 1 
        else:
            raise Exception("No such shape, please check the input")

    def change_conductivity(self, element_list, resistance_list):
        """

        Change conductivity in certain area according to ELEMENT NUMBER

        """
        if len(element_list) == len(resistance_list):
            for i, ele_num in enumerate(element_list):
                if ele_num > self.elem_num:
                    raise Exception("Element number exceeds limit")
                self.elem_perm[ele_num] = resistance_list[i]
        else:
            raise Exception('The length of element doesn\'t match the length of capacity')
    
    def calculate_FEM(self, theta):
        # changing theta could help increasing the accuracy
        potential_f = np.zeros((self.node_num_f , 1) , dtype=np.complex128) # set the phi_f and phi_b
        potential_b = (np.cos(theta) + 1j * math.sin(theta)) * np.ones((self.node_num_bound , 1))
        K_f = self.K_sparse[0 : self.node_num_f, 0 : self.node_num_f]
        K_b = self.K_sparse[0 : self.node_num_f, self.node_num_f : self.node_num]
        #since = time.time()
        potential_f = calculate_FEM_equation(potential_f, K_f, K_b, potential_b)
        #potential_f = - np.dot(np.dot(np.linalg.inv(K_f) , K_b) , potential_b) #solving the linear equation set
        #print(time.time()-since)
        """
        K_f_gpu = cp.asarray(K_f)
        K_b_gpu = cp.asarray(K_b)
        potential_b_gpu = cp.asarray(potential_b)
        potential_f_gpu = - cp.dot(cp.dot(cp.linalg.inv(K_f_gpu) , K_b_gpu) , potential_b_gpu) #solving the linear equation set
        potential_f = cp.asnumpy(potential_f_gpu)
        potential_b = cp.asnumpy(potential_b_gpu)
        """
        potential_f = np.reshape(potential_f , (-1))
        potential_b = np.reshape(potential_b , (-1))
        potential_f = np.append(potential_f , potential_b)
        return potential_f

    def sync_back_potential(self):
        """
        Put the potential back in order
        """
        potential = np.copy(self.node_potential)
        for i, j in enumerate(self.K_node_num_list):
            self.node_potential[j] = potential[i] 

    def calculate_element_potential(self):
        """
        Get of each element potential
        Average of each nodes
        """
        for i, _ in enumerate(self.elem_param):
            k1 , k2 , k3 = self.elem[i]
            self.element_potential[i] = (self.node_potential[k1]+self.node_potential[k2]+self.node_potential[k3])/3

    def calc_electrode_elements(self, electrode_number, center, radius):
        """
        Get the electrode element sets for every electrode 
        According to the square area given and put values into electrode_mesh dict
        """
        if electrode_number >= self.electrode_num:
            raise Exception("the input number exceeded electrode numbers")
        else:
            center_x, center_y = center
            count = 0
            for i, x in enumerate(self.elem_param[:,7]):
                if x <= (center_x + radius) and x >= (center_x - radius) and self.elem_param[i][8] <= (center_y + radius) and self.elem_param[i][8] >= (center_y - radius):
                    self.electrode_mesh[electrode_number].append(i)
                    count += 1
            if count == 0:
                raise Exception("No element is selected, please check the input")
    
    def calc_electrode_potential(self):
        """
        Get the mean value of potential on every electrode
        """
        for i, elements in enumerate(self.electrode_mesh.values()):
            potential = []
            for element in elements:
                potential.append(self.element_potential[element])
            self.electrode_potential[i] = np.mean(np.array(potential))
    
    def swap(self, a, b):
        """
        Swap two rows and columns of the sparse matrix
        """
        self.K_sparse[[a,b], :] = self.K_sparse[[b,a], :]
        self.K_sparse[:, [a,b]] = self.K_sparse[:, [b,a]]
        self.K_node_num_list[a],self.K_node_num_list[b] = self.K_node_num_list[b], self.K_node_num_list[a]

@jit
def calculate_FEM_equation(potential_f, K_f, K_b, potential_b):
    return - np.dot(np.dot(np.linalg.inv(K_f) , K_b) , potential_b) #solving the linear equation set


