#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 18:19:58 2023

@author: solene
"""


####This sccript takes as inputs the spectral decomposition and the diffusion communities made at time tau
####It outputs the <pin>_C(t) and <pout>_C(t) in a dataframe

import numpy as np
import scipy
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
from scipy import sparse
from sklearn.cluster import KMeans
import time
import pandas as pd
import mat73
from scipy.io import savemat, loadmat




####fill the paths for the network of interest
str_network='config_model_2p8'
path_spectral="/home/solene/Documents/Solene/DataMarseille/ganglion_lymph/paper_draft4/diffusion_pipeline/repository_github/Spectral_decompositions/"
path_com='/home/solene/Documents/Solene/DataMarseille/ganglion_lymph/paper_draft4/diffusion_pipeline/repository_github/Diffusion_communities/'
path_save='/home/solene/Documents/Solene/DataMarseille/ganglion_lymph/paper_draft4/diffusion_pipeline/repository_github/output/'
####chose the times list
#ts=[100,250,500,750,1000,2000,5000,tau,20000,50000] ####PVor
#ts=[100,250,500,750,1000,2000,tau, 10000,20000,50000]####HVor
#ts=[100,200,500,750,1000,2000,5000,10000,tau,50000]###LN
#ts=[20,50,100,tau,250,500,1000] ###HVor_rewired_2
ts=[1,2,5,10,20,50,100]###config model 2p8
#ts=[935]####LN_bio


####load spectral decomposition and communities
psi=scipy.sparse.load_npz(path_spectral+'config_model_2p8_psis.npz')
phi=scipy.sparse.load_npz(path_spectral+'config_model_2p8_phis.npz')
Lambda=scipy.sparse.load_npz(path_spectral+'config_model_2p8_Lambda.npz')
Lambda=np.real(Lambda)
idx=np.load(path_com+'clusters_'+str_network+'.npy')


#####find relaxation time
tau=1/(1-Lambda[1,1])
tau=np.round(tau)
print( 'relaxation time is '+str(tau))


k=100


def make_Lambda_t(Lambda, t):
    Lambda_t=np.zeros((2000,2000))
    for i in range(2000):
        Lambda_t[i,i]=np.power(Lambda[i,i],t)
    Lambda_t=csr_matrix(Lambda_t)   
    return Lambda_t

def find_pin(idx,ic,psi,phi,Lambda_t):
    ind=np.where(idx==ic)[0]
    not_ind=np.where(idx!=ic)[0]
    N=np.shape(psi)[0]
    u=np.zeros((N,1))
    N_ind=np.shape(ind)[0]
    u[ind,:]=1/N_ind
    v=psi.dot(Lambda_t.dot(np.transpose(phi).dot(u)))
    v_in=v[not_ind,:]
    p_in=np.mean(v_in)
    std_in=np.std(v_in)
    return p_in, std_in

def find_pout(idx,ic,psi,phi,Lambda_t):
    ind=np.where(idx==ic)[0]
    not_ind=np.where(idx!=ic)[0]
    N=np.shape(psi)[0]
    u=np.zeros((N,1))
    N_not_ind=np.shape(not_ind)[0]
    u[not_ind,:]=1/N_not_ind
    v=psi.dot(Lambda_t.dot(np.transpose(phi).dot(u)))
    v_out=v[ind,:]
    p_out=np.mean(v_out)
    std_out=np.std(v_out)
    return p_out, std_out



df_pinpout=pd.DataFrame(columns=['time','ic','pin','pout','std_in','std_out'])
for t in ts:
    print('t='+str(t))
    start_time = time.time()
    print("i am starting poinpout for "+str_network)
    Lambda_t=make_Lambda_t(Lambda,t)
    for ic in range(k):
        print('cluster number '+str(ic))
        [p_in,std_in]=find_pin(idx,ic,psi,phi,Lambda_t)
        [p_out,std_out]=find_pout(idx,ic,psi,phi,Lambda_t)
        line=pd.DataFrame({'time':[t],'ic':[ic],'pin':[p_in], 'pout':[p_out], 'std_in':[std_in], 'std_out':[std_out]})
        df_pinpout=pd.concat([df_pinpout,line])
    print("--- %s seconds ---" % (time.time() - start_time))
    print('done with pinpout')
df_pinpout.to_pickle(path_save+'pinpout_all_t_'+str_network+'.pkl')