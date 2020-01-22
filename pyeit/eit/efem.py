from __future__ import division, absolute_import, print_function

import numpy as np 
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from scipy import sparse
import math



class Forward(object):
    """
    FEM solving for capacity included forward problem
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
    
    electrode_meshes: electrode_numbers NdArray
    
    frequency : float Input frequency

    """

    def __init__(self, mesh, electrode_meshes = np.array([[0]]), freqency = 20000.0):
        """parameters
        mesh: dict
             mesh structure
             2D Mesh triangle elements 
        electrode_meshes: 2D array 
            electrode mesh element numbers 
        """
        self.nodes = mesh['node']
        self.node_num = np.shape(self.nodes)[0]
        self.node_num_bound = 0
        self.node_num_f = 0
        self.elem =  mesh['element']
        self.elem_num = np.shape(self.elem)[0]
        self.elem_perm = mesh['perm']
        self.elem_capacity = np.zeros(np.shape(self.elem_perm))
        self.electrode_mesh = electrode_meshes
        self.electrode_nums = np.shape(self.electrode_mesh)[0]
        self.elem_param = np.zeros((np.shape(self.elem)[0],7)) # area, b1, b2, b3, c1, c2, c3
        self.freq = freqency
        self.K_sparse = np.zeros((self.node_num,self.node_num),dtype = np.complex128)
        self.K_node_num_list = [x for x in range(self.node_num)]# Node number mapping list when calculating
        self.node_potential = np.zeros((self.node_num), dtype = np.complex128)
        self.element_potential = np.zeros((self.elem_num),dtype = np.complex128)
        
    def calculation(self):
        self.calculate_param()
        self.construct_sparse_matrix()
        self.set_boundary_condition()
        #split time into a  
        for i in range(100):
            theta = 2*math.pi/100 * i
            self.node_potential += np.abs(self.calculate_FEM(theta))
        self.node_potential /= 100
        self.calculate_element_potential()
        return self.node_potential, self.element_potential
    
    def JACcalculation(self):
        """
        calculate Jaccobian matrix for Amplitude and phase
        """

    def calculate_param(self):
        """
        Update parameters for each element
        
        """
        x = [.0,.0,.0]
        a = [.0,.0,.0]
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
            self.elem_param[count] = [area, b[0], b[1], b[2], c[0], c[1], c[2]]
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

    def set_boundary_condition(self):
        """
        Update boundary condition according to electrode mesh
        The boundary is at the input electrode whose potential is all Ae ^ iPhi

        And swap the index of matrix put the boundary elements at the bottom of the sparse matrix
        """
        node_list = np.reshape(self.electrode_mesh, (-1)) # reshape to 1D
        node_list = list(np.unique(node_list)) # get rid of repeat numbers 
        index = self.node_num
        self.node_num_bound = len(node_list)
        self.node_num_f = self.node_num - self.node_num_bound
        for list_num in node_list:
            index = index - 1
            self.swap(list_num, index)

    def change_capacity(self, element_list, capacity_list):
        """
        """
        if len(element_list) == len(capacity_list):
            for i, ele_num in enumerate(element_list):
                self.elem_capacity[ele_num] = capacity_list[i]
        else:
            print('The length of element doesn\'t match the length of capacity')
    
    def change_conductivity(self, element_list, resistance_list):
        if len(element_list) == len(resistance_list):
            for i, ele_num in enumerate(element_list):
                self.elem_perm[ele_num] = resistance_list[i]
        else:
            print('The length of element doesn\'t match the length of capacity')


    def calculate_FEM(self, theta):
        theta = math.pi / 4 # changing theta could help increasing the accuracy
        potential_f = np.zeros((self.node_num_f , 1) , dtype=np.complex128) # set the phi_f and phi_b
        potential_b = (math.cos(theta) + 1j * math.sin(theta)) * np.ones((self.node_num_bound , 1))
        K_f = self.K_sparse[0 : self.node_num_f, 0 : self.node_num_f]
        K_b = self.K_sparse[0 : self.node_num_f, self.node_num_f : self.node_num]
        potential_f = - np.dot(np.dot(np.linalg.inv(K_f) , K_b) , potential_b) #solving the linear equation set
        potential_f = np.reshape(potential_f , (-1))
        potential_b = np.reshape(potential_b , (-1))
        potential_f = np.append(potential_f , potential_b)
        return potential_f

    def calculate_element_potential(self):
        for i, element_param in enumerate(self.elem_param):
            k1 , k2 , k3 = self.elem[i]
            self.element_potential[i] = (self.node_potential[k1]+self.node_potential[k2]+self.node_potential[k3])/3


        
    def swap(self, a, b):
        """
        Swap two rows and columns of the sparse matrix
        """
        self.K_sparse[[a,b], :] = self.K_sparse[[b,a], :]
        self.K_sparse[:, [a,b]] = self.K_sparse[:, [b,a]]
        self.K_node_num_list[a],self.K_node_num_list[b] = self.K_node_num_list[b], self.K_node_num_list[a]

    
