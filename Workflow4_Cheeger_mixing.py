#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 18:35:04 2023

@author: solene
"""




####This sccript takes as inputs the spectral decomposition and the diffusion communities made at time tau
####It outputs the Cheeger mixing for each community

import numpy as np
import scipy
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
from scipy import sparse
from sklearn.cluster import KMeans
import time
import mat73
from scipy.io import savemat, loadmat



str_network='config_model_2p8'
path_adj="/home/solene/Documents/Solene/DataMarseille/ganglion_lymph/paper_draft4/diffusion_pipeline/repository_github/Adjacency_matrices/"
path_com='/home/solene/Documents/Solene/DataMarseille/ganglion_lymph/paper_draft4/diffusion_pipeline/repository_github/Diffusion_communities/'
path_save='/home/solene/Documents/Solene/DataMarseille/ganglion_lymph/paper_draft4/diffusion_pipeline/repository_github/output/'
A=scipy.sparse.load_npz(path_adj+'A_config_model_2p8.npz')
idx=np.load(path_com+'clusters_'+str_network+'.npy')

def find_Cheeger_mix(A,idx,ic,d):
    ind=np.where(idx==ic)[0]
    not_ind=np.where(idx!=ic)[0]
    sub_A=A[ind,:]
    sub_A=sub_A[:,not_ind]
    E_inter=np.sum(sub_A)
    V_ind=np.sum(d[ind])
    V_not_ind=np.sum(d[not_ind])
    hG=E_inter/np.min([V_ind,V_not_ind])
    return hG

###########################################
g=nx.from_numpy_array(A)
d=nx.degree(g)
d=np.array(d)
d=d[:,1]


#######################################################
start_time = time.time()
print('starting analysis of '+str_network)
k=100
hGs=np.zeros((k,1))
for ic in range(k):
    hG=find_Cheeger_mix(A,idx,ic,d)
    hGs[ic]=hG
np.save(path_save+'hGs_'+str_network+'.npy',hGs)

