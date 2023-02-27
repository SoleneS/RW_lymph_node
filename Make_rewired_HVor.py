#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:26:24 2023

@author: solene
"""



import numpy as np
import scipy.io
import networkx as nx
from scipy.sparse import csr_matrix, save_npz
import matplotlib.pyplot as plt
import time
import copy
path_adj='/home/solene/Documents/Solene/DataMarseille/ganglion_lymph/paper_draft4/diffusion_pipeline/repository_github/Adjacency_matrices/'
path_save='/home/solene/Documents/Solene/DataMarseille/ganglion_lymph/paper_draft4/diffusion_pipeline/repository_github/output/'
A_HVor=scipy.sparse.load_npz(path_adj+'A_HVor.npz')
G=nx.from_numpy_array(A_HVor)
Ne=G.number_of_edges()

 
pc=[2,5,10,20]####percent rewiring
for p in pc:
    start_time = time.time()
    new_G=copy.deepcopy(G)
    Ne_rewire=int(np.round((p/100)*Ne))
    dummy=nx.connected_double_edge_swap(new_G,Ne_rewire)
    A=nx.adjacency_matrix(new_G)
    save_npz(path_save+'A_HVor_rewired_'+str(p)+'.npz',A)
    print("--- %s seconds ---" % (time.time() - start_time))




        

