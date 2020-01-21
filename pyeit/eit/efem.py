from __future__ import division, absolute_import, print_function

import numpy as np 
from scipy import sparse
import math



class Forward(object):
    """
    FEM solving for capacity included forward problem
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
        self.node_num = self.nodes.shape(0)
        self.node_num_bound = 0
        self.node_num_f = 0
        self.elem = mesh['element']
        self.elem_perm = mesh['perm']
        self.elem_capacity = np.zeros(self.elem_perm.shape())
        self.electrode_mesh = electrode_meshes
        self.electrode_nums = self.electrode_mesh.shape(0)
        self.elem_param = np.zeros((self.elem.shape(0),7)) # area, b1, b2, b3, c1, c2, c3
        self.freq = freqency
        self.K_sparse = np.zeros((self.node_num,self.node_num),dtype = np.complex128)
        self.K_node_num_list = [x for x in range(self.node_num)]# Node number mapping list when calculating
        self.node_potential = np.zeros((self.node_num), dtype = np.complex128)
        
    def calculation(self):
        self.calculate_param()
        self.construct_sparse_matrix()
        self.set_boundary_condition()
        self.calculate_FEM()


    def calculate_param(self):
        """
        Update parameters for stiffness matrix
        """
        x = y = a = b = c = [.0,.0,.0]
        count = 0
        for element in self.elem:
            x[0], y[0] = self.nodes[element[0]]
            x[1], y[1] = self.nodes[element[1]]
            x[2], y[2] = self.nodes[element[2]]
            for i in range(3):
                a[i] = x[(1+i)%3] * y[(2+i)%3] - x[(2+i)%3] * y[(1+i)%3]
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
        patern = [[0,0],[1,1],[2,2],[0,1],[1,2],[2,0]]
        for element in self.elem:
            param = self.elem_param[index]
            for i,j in patern:
                if self.K_sparse[element[i]][element[j]] == 0:
                    if i != j :
                        # stiffness k_ij = sigma/4 * (bk1*bk2 + ck1*ck2) - j * w * capacity * (bk1 * ck2 - bk2 * ck1) /24
                        K_ij = self.elem_perm[index] * (param[1+i] * param[1+j] + param[4+i] * param[4+j]) / 4 - (self.freq * self.elem_capacity * param[0] /12) * 1j 
                    else:
                        K_ij = self.elem_perm[index] * (param[1+i] * param[1+j] + param[4+i] * param[4+j]) / 4 - (self.freq * self.elem_capacity * param[0] /6) * 1j  
                    self.K_sparse[element[i]][element[j]] = self.K_sparse[element[j]][element[i]] = K_ij
            index += 1

    def set_boundary_condition(self):
        """
        Update boundary condition according to electrode mesh
        And swap the index of matrix
        """
        node_list = np.reshape(self.electrode_mesh, (-1)) # reshape to 1D
        node_list = list(np.unique(node_list)) # get rid of repeat numbers 
        index = self.node_num
        self.node_num_bound = len(node_list)
        self.node_num_f = self.node_num - self.node_num_bound
        for list_num in node_list:
            index = index - 1
            self.swap(list_num, index)

    def change_capacity(self, element_dict):
        


    def calculate_FEM(self):
        theta = math.pi / 4 # changing theta could help increasing the accuracy
        potential_f = np.zeros((1,self.node_num_f),dtype=np.complex128) # set the phi_f and phi_b
        potential_b = (math.cos(theta) + 1j * math.sin(theta)) * np.ones((1,self.node_num_bound))
        K_f = self.K_sparse[[0:self.node_num_f - 1], [0:self.node_num_f - 1]]
        K_b = self.K_sparse[[0 : self.node_num_f], [self.node_num_f : self.node_num - 1]]
        potential_f = -np.dot(np.dot(np.linalg.inv(K_f),K_b),potential_b)
        potential_f = np.reshape(potential_f,(-1))
        potential_b = np.reshape(potential_b,(-1))
        np.append(potential_f,potential_b)
        self.node_potential = potential_f[self.K_node_num_list] #change back to mesh node No.




        
    def swap(self, a, b):
        """
        Swap two rows and columns of the sparse matrix
        """
        self.K_sparse[[a,b], :] = self.K_sparse[[b,a], :]
        self.K_sparse[:, [a,b]] = self.K_sparse[:, [b,a]]
        self.K_node_num_list[a,b] = self.K_node_num_list[b,a]

    
