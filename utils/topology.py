#transforme les signaux ecog en complexes simpliciaux

import numpy as np
import torch
import networkx as nx
import scipy.sparse as sp
import toponetx as tnx

#transforme les données ecog en matrice de distance 1-cor

def ecog_to_distance_matrix(X):
    batch,T,d=X.shape
    D=np.zeros((batch,d,d),dtype=np.float32)
    for b in range(batch):
        xb=X[b, -375: ]
        corr=np.corrcoef(xb.T)
        
        D[b]=np.clip(1.0-np.abs(corr), 0,1.0) #valeur ABSOLUE 
        D[b]=(D[b]+D[b].T)/2.0
        np.fill_diagonal(D[b],0.0)
    return D

#ici j'ai la distance calculer en haut en complexe de ribs
def distance_matrix_to_rips_complex(D,r=0.8,max_dim=2):
    #arete (i,j) ajoutee si D[i][j]<=r
    #triangle (i,j,k) ajoute si les 3 aretes sont la
    d=D.shape[0]
    G=nx.Graph()
    G.add_nodes_from(range(d))
    for i in range(d):
        for j in range(i+1,d):
            G.add_edge(i,j,weight=float(D[i,j]))
    return tnx.weighted_graph_to_vietoris_rips_complex(G,r=r,max_dim=max_dim)

def batch_to_rips_complexes(D_batch,r=0.8,max_dim=2):

    #applique rips sur tout le batch
    return [distance_matrix_to_rips_complex(D_batch[b],r=r,max_dim=max_dim) for b in range(len(D_batch))]


def _sp_to_torch(M):
    #sparse scipy,tenseur pytorch dense
    return torch.tensor(M.toarray(),dtype=torch.float32)


def get_matrices_from_sc(sc):

    #extrait B1,B2,L0,L1d,L1u,L2 depuis un complexe simplicial
    n_nodes=len(list(sc.nodes))

    #B1 matrice d incidence noeuds x aretes
    B1_sp=sc.incidence_matrix(rank=1) if sc.dim>=1 else None
    n_edges=B1_sp.shape[1] if B1_sp is not None else 0
    B1=_sp_to_torch(B1_sp) if B1_sp is not None else torch.zeros(n_nodes,0)
    #B2 aretes x faces (vide si pas de triangles)
    B2_sp=sc.incidence_matrix(rank=2) if sc.dim>=2 else None
    n_faces=B2_sp.shape[1] if B2_sp is not None else 0
    B2=_sp_to_torch(B2_sp) if B2_sp is not None else torch.zeros(max(n_edges,1),0)

    #laplaciens de hodge
    L0=B1@B1.T
    L1d=B1.T@B1
    L1u=B2@B2.T
    L2=B2.T@B2

    return {
        "n_nodes":n_nodes,"n_edges":n_edges,"n_faces":n_faces,
        "B1":B1,"B2":B2,
        "L0":L0,"L1":L1d+L1u,"L1d":L1d,"L1u":L1u,"L2":L2,
        #features initiales: que des 1, le modele apprend depuis la structure topo
        "x0":torch.ones(n_nodes,1),
        "x1":torch.ones(n_edges,1),
        "x2":torch.ones(n_faces,1),
    }
