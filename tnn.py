#tnn.py
#modele SCNN pour classifier les epochs ECoG en 0 ou 1

import torch
import torch.nn as nn
import torch.nn.functional as F
from topomodelx.nn.simplicial.scnn_layer import SCNNLayer


#SCNN travaille uniquement sur les aretes
#pour chaque arete on agrege l info de ses voisins via deux laplaciens:
#L1d = voisins par le bas (aretes qui partagent un noeud)
#L1u = voisins par le haut (aretes qui partagent un triangle )
#a la fin mean pool sur toutes les aretes 

class SCNNClassifier(nn.Module):

    def __init__(self,in_channels=1,hidden_channels=32,n_layers=2,conv_order_down=2,conv_order_up=2,dropout=0.3):
        super().__init__()
        self.layers=nn.ModuleList()
        #premiere couche in a hidden, les suivantes hidden ahidden
        self.layers.append(SCNNLayer(in_channels=in_channels,out_channels=hidden_channels, conv_order_down=conv_order_down,conv_order_up=conv_order_up))
        for _ in range(n_layers-1):
            self.layers.append(SCNNLayer(in_channels=hidden_channels,out_channels=hidden_channels,conv_order_down=conv_order_down,conv_order_up=conv_order_up))
        self.dropout=nn.Dropout(dropout)
        self.classifier=nn.Linear(hidden_channels,1)

    def forward_single(self,mats):
        x1=mats["x1"]    #features des aretes, des 1 au debut
        L1d=mats["L1d"]  #laplacien bas
        L1u=mats["L1u"]  #laplacien haut
        for layer in self.layers:
            x1=self.dropout(F.relu(layer(x1,L1d,L1u)))
        #mean pool vecteur fixe peu importe le nb d aretes
        return self.classifier(x1.mean(dim=0)).squeeze()

    def forward(self,batch_mats):
        #on traite chaque sample un par un car tailles variables
        return torch.stack([self.forward_single(m) for m in batch_mats])


def build_model(**kwargs):
    return SCNNClassifier(**kwargs)