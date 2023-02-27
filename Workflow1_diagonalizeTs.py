#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 11:46:10 2023

@author: solene
"""

####This script takes as input the adjacency matrix of the graph 
####of interest and outputs its approximated spectral decomposition 
####Psi, Lambda, Phi, with approximation cut-off k=2000



import numpy as np
import scipy
from scipy.sparse.linalg import eigsh
from scipy.io import loadmat
from scipy.io import savemat
import time
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
from scipy import sparse


####load the adjacency matrix, input to fill
path_graph="/home/solene/Documents/Solene/DataMarseille/ganglion_lymph/paper_draft4/diffusion_pipeline/repository_github/Adjacency_matrices/"
A=sparse.load_npz(path_graph+"A_LN.npz")
name='LN' 
path_save='/home/solene/Documents/Solene/DataMarseille/ganglion_lymph/paper_draft4/diffusion_pipeline/repository_github/output/'

####make the symmetrized transition matrix Ts to diagonalize
N=np.shape(A)[0]
G=nx.from_numpy_array(A)
deg=np.sum(A,1)
D=csr_matrix((N,N))
D_inv=csr_matrix((N,N))
D_12=csr_matrix((N,N))
D_12_inv=csr_matrix((N,N))
for i in range(N):
    d=deg[i]
    D[i,i]=d
    D_inv[i,i]=1/d
    D_12[i,i]=np.sqrt(d)
    D_12_inv[i,i]=1/(np.sqrt(d))
T=D_inv.dot(A)
Ts=D_12.dot(T.dot(D_12_inv))


####diagonalization, takes an overnight computation time 
def diagonalize(Ts,name):
    print("i am starting diagonalizing "+name)
    start_time = time.time()
    vals, vecs=scipy.sparse.linalg.eigsh(Ts, k=2000, M=None, sigma=None, which='LM')
    print("--- %s seconds ---" % (time.time() - start_time))
    return vals, vecs


vals,vecs=diagonalize(Ts,name)
vals_sparse=csr_matrix(vals)
vecs_sparse=csr_matrix(vecs)


####save the eigenvalues and eigenvectors
sparse.save_npz(path_save+name+"_vals.npz",vals_sparse)
sparse.save_npz(path_save+name+"_vecs.npz", vecs_sparse)


#####sort by decreasing values of eigenvalues
vals_abs=np.abs(vals)
ind=vals_abs.argsort()
ind_flip=np.flipud(ind)
vals_sorted=vals[ind_flip]
vecs_sorted=vecs[:,ind_flip]

   
###############make Psi, Lambda, Phi
Lambda=np.diag(vals_sorted)
if np.sign(vecs_sorted[0,0])==-1:
    V=-vecs_sorted
else:
    V=vecs_sorted

Dnorm12=csr_matrix((N,N))
Dnorm_minus12=csr_matrix((N,N))

for i in range(N):
    Dnorm12[i,i]=np.sqrt(D[i,i]/np.sum(D))
    Dnorm_minus12[i,i]=np.sqrt(np.sum(D)/D[i,i])

V=csr_matrix(V)
phis=np.dot(Dnorm12,V)
psis=np.dot(Dnorm_minus12,V)
Lambda=csr_matrix(Lambda)
#################save Psi, Lambda, Phi
sparse.save_npz(path_save+name+'_psis.npz', psis)
sparse.save_npz(path_save+name+'_phis.npz', phis)
sparse.save_npz(path_save+name+'_Lambda.npz', Lambda)



