# RW_lymph_node
Anaysis workflow for the paper "Random walk informed heterogeneities detection reveals how the lymph node conduits network influences T-cells collective exploration behavior"

There are 4 scripts to generate the null networks
- Make_config_model_2p8.py
- Make_HVor.py
- Make_PVor.py
- Make_rewired_HVor.py

and 4 scripts that run the steps of the workflow to make and analyze the diffusion communities
- Workflow1_diagonalizeTs.py
- Workflow2_make_diffusion_communities.py
- Workflow3_pinpout_time.py
- Workflow4_Cheeger_mixing.py

The inputs and outputs of the scripts used in the results of the paper are stored into the folders "Adjacency_matrices", "Diffusion_communities", "Pinpout_time", and "Cheeger_mixing" in this repositoty, and the spectral decompositions (left and right eigenvectors, eigenvalues diagonal matrices) of the transition matrix generated by Workflow1_diagonalizeTs.py are stored in zenodo DOI: 10.5281/zenodo.7681900
https://zenodo.org/record/7681900#.Y_8UOS8w1hF. In the script, the outputs are stored temporarily into the folder "output"

In addition there are 3 html files which correspond to the 3D visualization of the Figures5 B,C,D of the paper. 

The scripts that generate the null networks:

-Make_config_model_2p8.py
This script first generates a network with the same number of nodes as the LNCN, with all nodes being of degree 3.
Then, it randomly removes edges until the mean degree is 2.8

-Make_HVor.py
This script generates first a grid of equally spaced points. Then, the points outside of a sphere 
are removed. Then, the points are slightly shifted randomly. These points are used as seeds to generate a Voronoi mesh.
Then, it randomly removes edges until the mean degree is 2.8

-Make_PVor.py
This script generates first two overlapping gaussian distributions of points of different standard deviations. These node are used as seeds to generate a Voronoi mesh. 
Then, it randomly removes edges until the mean degree is 2.8.

-Make_rewired_HVor.py
This script departs from the HVor network, and rewires a certain increasing percentage of edges.


The analysis pipeline:

-Workflow1_diagonalizeTs.py
This script gives the spectral decomposition of the symmetrized transition matrix. 
It takes about 10h-13h to run on a computer with ...
input: adjacency matrices, in "Adjacency_matrices"
output:Psi, Lambda, Phi, saved  in zenodo archive https://zenodo.org/record/7681900#.Y_8UOS8w1hF

-Workflow2_make_diffusion_communities.py
This script makes 100 diffusion communities by kmeans in the diffusion space
input: Psi, Lambda, Phi, saved in "Spectral_decompositions"
output: the list of ids for 100 diffusion clusters for each node, stored in "Diffusion communities"

-Workflow3_pinpout_time.py
This script computes the <pin>C(t) and <pout>C(t) for each of the 100 diffusion communities, at different times
input: Psi, Lambda, Phi, saved in "Spectral_decompositions" and the clusters id in "Diffusion communities"
output: pinpout_all_t, a DataFrame that stores the <pin>C, <pout>C, for all the communities, at various times, stored in "Pinpout_time"

-Workflow4_Cheeger_mixing.py
This script computes the Cheeger mixing values for each of the 100 communities.
input: Psi, Lambda, Phi, saved in "Spectral_decompositions" and the clusters id in "Diffusion communities"
output: the list of Cheeger mixing value for each of the 100 communities, stored in "Cheeger_mixing"
