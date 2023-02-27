#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 18:10:27 2023

@author: solene
"""
####This script takes as input the approximated spectral decomposition, make the diffusion coordinates
####and outputs the kmeans clusters in the diffusion space, ie the diffusion communities


import numpy as np
import scipy
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
from scipy import sparse
import time
import faiss

####to fill, to load the spectral decomposition of the graph of interest
str_network='config_model_2p8'
path_spectral="/home/solene/Documents/Solene/DataMarseille/ganglion_lymph/paper_draft4/diffusion_pipeline/repository_github/Spectral_decompositions/"
psi=scipy.sparse.load_npz(path_spectral+'config_model_2p8_psis.npz')
phi=scipy.sparse.load_npz(path_spectral+'config_model_2p8_phis.npz')
Lambda=scipy.sparse.load_npz(path_spectral+'config_model_2p8_Lambda.npz')
Lambda=np.real(Lambda)
path_save='/home/solene/Documents/Solene/DataMarseille/ganglion_lymph/paper_draft4/diffusion_pipeline/repository_github/output/'
print('done with loading variables')


#####find relaxation time
tau=1/(1-Lambda[1,1])
tau=np.round(tau)
print( 'relaxation time is '+str(tau))

#####fill the matrix Lambda^tau
Lambda_tau=np.zeros((2000,2000))
for i in range(2000):
    Lambda_tau[i,i]=np.power(Lambda[i,i],tau)
Lambda_tau=csr_matrix(Lambda_tau)   
print('done filling Lambda_tau')

####create diffusion coordinates at time tau
Dcoord=psi.dot(Lambda_tau)
print('done with creating the diffusion coord')

####make kmeans clusters
start_time = time.time()
k=100
max_iter=300
kmeans=faiss.Kmeans(np.shape(Dcoord)[1],k=k,niter=max_iter, nredo=1)
Dcoord=Dcoord.todense()
kmeans.train(Dcoord.astype(np.float32))
print("--- %s seconds ---" % (time.time() - start_time))
idx=kmeans.index.search(Dcoord.astype(np.float32), 1)[1]
np.save(path_save+'clusters_'+str_network+'.npy',idx)
