#train.py
#moteur d entrainement pour le pipeline ECoG-TNN

import time
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,confusion_matrix,classification_report
from ecog_tnn.models.tnn import build_model


@dataclass
class TrainerConfig:
    #tous les hyperparamètres en un seul endroit
    in_channels: int =  32
    hidden_channels: int = 32
    out_channels : int = 32
    n_layers: int = 2
    conv_order_down: int = 2
    conv_order_up: int = 2
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 50
    patience: int = 15       #nb d epochs sans amelioration avant d arreter
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pos_weight: Optional[float] = None  #si les classes sont desequilibrees
    seed: int = 42


class Trainer:

    def __init__(self,cfg:TrainerConfig):
        self.cfg=cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        #on utilise uniquement scnn
        self.model=build_model(
            in_channels=cfg.in_channels,
            hidden_channels=cfg.hidden_channels,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            conv_order_down=cfg.conv_order_down,
            conv_order_up=cfg.conv_order_up,
        ).to(cfg.device)

        pw=torch.tensor([cfg.pos_weight],device=cfg.device) if cfg.pos_weight else None
        self.criterion=nn.BCEWithLogitsLoss(pos_weight=pw)

        # Adam avec weight decay decouple pour regulariser
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)

        #cosine : lr descend en forme de cosinus, doux et efficace
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=cfg.n_epochs)

        #historique pour tracer les courbes apres
        self.history={"train_loss":[],"val_loss":[],"val_acc":[],"val_auc":[]}
        self.best_val_auc=-1.0
        self.best_state=None  #poids du meilleur modele en memoire

    def _run_epoch(self,loader,train):
        self.model.train(train)
        total_loss=0.0
        all_logits,all_labels=[],[]

        for batch_mats,labels in loader:
            labels=labels.to(self.cfg.device)
            if train:
                self.optimizer.zero_grad()
            logits=self.model(batch_mats)
            loss=self.criterion(logits,labels)
            if train:
                loss.backward()
                #gradient clipping empeche les gradients d exploser
                nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
                self.optimizer.step()
            total_loss+=loss.item()*len(labels)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.cpu())

        all_logits=torch.cat(all_logits).numpy()
        all_labels=torch.cat(all_labels).numpy().astype(int)
        all_preds=(all_logits>0.0).astype(int)  #seuil a 0 car pas encore sigmoid
        avg_loss=total_loss/len(all_labels)
        acc=accuracy_score(all_labels,all_preds)
        try:
            auc=roc_auc_score(all_labels,all_logits)
        except ValueError:
            auc=0.5  #si toutes les preds sont du meme cote
        return avg_loss,acc,auc

    def fit(self,train_loader,val_loader):
        patience_counter=0
        t0=time.time()
        for epoch in range(1,self.cfg.n_epochs+1):
            tr_loss,tr_acc,tr_auc=self._run_epoch(train_loader,train=True)
            vl_loss,vl_acc,vl_auc=self._run_epoch(val_loader,train=False)
            self.scheduler.step()
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(vl_loss)
            self.history["val_acc"].append(vl_acc)
            self.history["val_auc"].append(vl_auc)

            #on garde les poids du meilleur modele selon la val_auc
            if vl_auc>self.best_val_auc:
                self.best_val_auc=vl_auc
                self.best_state={k:v.cpu().clone() for k,v in self.model.state_dict().items()}
                patience_counter=0
            else:
                patience_counter+=1

            if epoch%5==0 or epoch==1:
                print(f"Epoch {epoch:3d}/{self.cfg.n_epochs}, tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}, vl_loss={vl_loss:.4f} vl_acc={vl_acc:.3f} vl_auc={vl_auc:.3f}, {time.time()-t0:.1f}s")

            #early stopping : si ca s ameliore plus depuis patience epochs on arrete
            if patience_counter>=self.cfg.patience:
                print(f"stop tot a l'epoch {epoch}")
                break

        #on recharge les poids du meilleur moment
        if self.best_state is not None:
            self.model.load_state_dict({k:v.to(self.cfg.device) for k,v in self.best_state.items()})
        print(f"meilleure val AUC: {self.best_val_auc:.4f}")
        return self.history

    def evaluate(self,loader):
        #evaluation finale sur le test set
        #desactive les gradients, plus rapide et moins de memoire
        self.model.eval()
        all_logits,all_labels=[],[]
        with torch.no_grad():
            for batch_mats,labels in loader:
                labels=labels.to(self.cfg.device)
                logits=self.model(batch_mats)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
        logits=torch.cat(all_logits).numpy()
        labels=torch.cat(all_labels).numpy().astype(int)
        preds=(logits>0.0).astype(int)
        proba=torch.sigmoid(torch.tensor(logits)).numpy()
        acc=accuracy_score(labels,preds)
        f1=f1_score(labels,preds,zero_division=0)
        try:
            auc=roc_auc_score(labels,logits)
        except ValueError:
            auc=0.5
        cm=confusion_matrix(labels,preds)
        report=classification_report(labels,preds,target_names=["class_0","class_1"])
        print(f"acc:{acc:.3f} f1:{f1:.3f} auc:{auc:.3f}")
        print("Matrice de confusion :",cm)
        print(report)
        return dict(accuracy=acc,f1=f1,auc_roc=auc,confusion_matrix=cm,report=report,logits=logits,proba=proba,labels=labels)