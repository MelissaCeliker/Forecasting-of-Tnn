# transforme les signaux ECoG en complexes simpliciaux pour pytorch
# appelles les fonctions de topology.py
# stock les resultat dans dataset que le dataloader peut utiliser et gere le decoupage en batch

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict

from ecog_tnn.utils.topology import (
    ecog_to_distance_matrix,
    batch_to_rips_complexes,
    get_matrices_from_sc,
)


class ECoGSimplicialDataset(Dataset):
    # pour chaque epoch on construit un complexe de Rips
    #labels binaires {0, 1}

    def __init__(self, X, y, r="0.8", max_dim=2):
        assert X.ndim == 3, "X doit etre de la forme (N, T, d)"
        assert len(X) == len(y)

        self.X = X
        self.y = torch.tensor(y, dtype=torch.float32)
        self.max_dim = max_dim

        #step 1  matrice de distances entre canaux
        print(f" DATASET : calcul des distances par correlation")
        self.D = ecog_to_distance_matrix(X)

        # etapes 2 choix du rayon de filtration
        self.r = float(r)

        # step 3  construction des complexes simpliciaux
        self.mats = None
        print(f" DATASET construction des complexes ")
        complexes = batch_to_rips_complexes(self.D, r=self.r, max_dim=max_dim)
        self.mats = [get_matrices_from_sc(sc) for sc in complexes]
        print(f" DATASET ok, {len(self.mats)} complexes construits")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # retourne le dict de matrices topo + le label
        return self.mats[idx],self.y[idx]



def simplicial_collate(batch):
    # collate custom car les complexes ont des tailles variables
    # on peut pas faire torch.stack sur des matrices (725, 725) et (1014, 1014)
    # donc on garde une liste de dicts, et on stack juste les labels
    mats_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return mats_list, labels


def build_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,r="auto", max_dim=2,batch_size=16, num_workers=0):
    train_ds = ECoGSimplicialDataset(X_train, y_train, r=r, max_dim=max_dim )
    val_ds = ECoGSimplicialDataset(X_val, y_val, r=train_ds.r, max_dim=max_dim)
    test_ds = ECoGSimplicialDataset(X_test, y_test, r=train_ds.r, max_dim=max_dim)
    kw = dict(collate_fn=simplicial_collate, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)
    print(f"LOAD OF DATA Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    return train_loader, val_loader, test_loader
