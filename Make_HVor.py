#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:16:39 2023

@author: solene
"""



import numpy as np
from scipy.spatial import Voronoi
import networkx as nx
from scipy import sparse
import copy

path_save='/home/solene/Documents/Solene/DataMarseille/ganglion_lymph/paper_draft4/diffusion_pipeline/repository_github/output/'
#####make grid
n=15
x=np.linspace(-n,n,2*n+1)
y=np.linspace(-n,n,2*n+1)
z=np.linspace(-n,n,2*n+1)
[X,Y,Z]=np.meshgrid(x,y,z)

N=np.shape(x)[0]
xyz=np.zeros((N**3,3))
i_fill=0
for i in range(N):
    print(i)
    for j in range(N):
        for k in range(N):
            xyz[i_fill,0]=X[i,j,k]
            xyz[i_fill,1]=Y[i,j,k]
            xyz[i_fill,2]=Z[i,j,k]
            i_fill=i_fill+1

#######make sphere inside grid
R=0.4*((np.sqrt(3)/2)*N)
dist=np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2);
xyz_sphere=xyz[dist<=R]

#######introduce some noise
rd=np.random.rand(N**3,3)
rd=rd/10
rel=np.random.randint(2,size=(N**3,3))
rel[rel[:,:]==0]=-1
noise=rd*rel
noised_xyz=xyz+noise

centers=noised_xyz
print('start to make Voronoi')
vor=Voronoi(centers)
print('done with making Voronoi')
N=np.shape(vor.vertices)
print('Number of vertices is'+str(N))


V=vor.vertices ###voronoi vertices
S=vor.ridge_vertices ####surface that separate two centers


E=[]
for s in S:
    n=np.shape(s)[0]
    for i in range(n-1):
        v0=s[i]
        v1=s[i+1]
        if (v0!=-1) and (v1!=-1):
            E.append((v0,v1))
    v0=s[n-1]
    v1=s[0]
    if (v0!=-1) and (v1!=-1):
        E.append((v0,v1))



G=nx.from_edgelist(E)
d=nx.degree(G)

####randomly remove edges until <d>=3

import copy
new_G=copy.deepcopy(G)
moy_d=4
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
    
    
A_Voronoi=nx.adjacency_matrix(new_G)
sparse.save_npz(path_save+"A_HVor.npz",A_Voronoi)