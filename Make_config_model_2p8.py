#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:41:42 2023

@author: solene
"""


#####make a regular d=3 graph using configuration model, then randomly remove edges until <d>=2.8

import numpy as np
import scipy.io
import networkx as nx
from scipy.sparse import csr_matrix, save_npz
from scipy import sparse
import scipy

N=192386
path_save='/home/solene/Documents/Solene/DataMarseille/ganglion_lymph/paper_draft4/diffusion_pipeline/repository_github/output/'
####list of degree for regular graph of degree 3 and same number of nodes as the LNCN
d_list=[]
for i in range(N):
    d_list.append(3)
    
####generate the regular graph and check it is connected    
G=nx.configuration_model(d_list)
if nx.is_connected(G):
    print('ok, it is connected')
####save the adjacency matrix
A=nx.adjacency_matrix(G)
save_npz(path_save+'A_config_model_reg_3.npz',A)



####list of edges
E=[]
for line in nx.generate_edgelist(G,data=False):
    print(line)
    line_split=line.split(' ')
    el1=int(line_split[0])
    el2=int(line_split[1])
    E.append((el1,el2))
        
####remove edges randomly until <d>=2.8  


from scipy import sparse
import scipy


d=nx.degree(G)

import copy
new_G=copy.deepcopy(G)
moy_d=3
count_not_connected=0
my_size=5000
while moy_d>2.8:
    new_E=list(new_G.edges)
    if count_not_connected>15:
        my_size=int(np.round(my_size/1.25))
        print('new size is '+str(my_size))
        count_not_connected=0
    r=np.random.randint(np.shape(new_E)[0],size=my_size)
    try_G=copy.deepcopy(new_G)
    rem = [new_E[index] for index in r]
    try_G.remove_edges_from(rem)
    if nx.is_connected(try_G):
        count_not_connected=0
        print(try_G.number_of_edges())
        new_G=try_G
        d=np.array(nx.degree(new_G))
        moy_d=np.mean(d[:,1])
        print(moy_d)
    else:
        count_not_connected=count_not_connected+1
        print('not connected')
    

A_config_model_2p8=nx.adjacency_matrix(new_G)
sparse.save_npz(path_save+"A_config_model_2p8.npz",A_config_model_2p8)