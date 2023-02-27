#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:31:26 2023

@author: solene
"""

import networkx as nx
import random
from scipy.io import savemat, loadmat
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
import copy
from scipy import sparse
path_save='/home/solene/Documents/Solene/DataMarseille/ganglion_lymph/paper_draft4/diffusion_pipeline/repository_github/output/'
n=28752
n1=10000
n2=n-n1

####two overlapping gaussian distributions of seeds
pos1 = {i: (random.gauss(0, 5), random.gauss(0, 5), random.gauss(0,5)) for i in range(n1)}
pos2 = {i: (random.gauss(5, 1), random.gauss(5, 1),random.gauss(5,1)) for i in range(n2)}
coord_array=np.array([[0]*3]*n)
coord_array=coord_array.astype(np.float)
i=0
for item_i in range(n1):
    my_tuple=pos1[item_i]
    x=my_tuple[0]
    y=my_tuple[1]
    z=my_tuple[2]
    coord_array[i,0]=x
    coord_array[i,1]=y
    coord_array[i,2]=z
    i=i+1

for item_i in range(n2):
    my_tuple=pos2[item_i]
    x=my_tuple[0]
    y=my_tuple[1]
    z=my_tuple[2]
    coord_array[i,0]=x
    coord_array[i,1]=y
    coord_array[i,2]=z
    i=i+1


####visualize how the seeds are distributed
df_coord=pd.DataFrame(coord_array,columns=['x','y','z'])
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

figure = go.Figure()

    
figure.add_trace(go.Scatter3d(x=df_coord.x,y=df_coord.y,z=df_coord.z,mode='markers',
                               marker=dict(size=1,opacity=0.5
                                                     )
                                        
                                         ))

          
figure.write_html(path_save+'centers_PVor.html')

####make voronoi graph

vor=Voronoi(coord_array)
V=vor.vertices ###voronoi vertices
S=vor.ridge_vertices 


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

####remove edges randomly until <d>=2.8

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
        print('not connected, current size is'+str(my_size))
    
    

A_Voronoi=nx.adjacency_matrix(new_G)
sparse.save_npz(path_save+"A_PVor.npz",A_Voronoi)