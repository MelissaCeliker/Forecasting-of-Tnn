#run_pipeline.py
#point d entree principal du pipeline ecog -> tnn

import argparse
import os
import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

from ecog_tnn.data.dataset import build_dataloaders
from ecog_tnn.experiments.train import Trainer,TrainerConfig

#in_channels tres haut, pareille hidden 

def run_pipeline(X,y,r="0.8",max_dim=2,hidden_channels=32,n_layers=2,n_epochs=50,batch_size=16,lr=1e-3,dropout=0.3,test_size=0.2,val_size=0.15,save_path=None,seed=42):

    #split stratifie: on garde les memes proportions de classes dans train/val/test
    X_trainval,X_test,y_trainval,y_test=train_test_split(X,y,test_size=test_size,random_state=seed,stratify=y, shuffle=True)
    val_frac=val_size/(1.0-test_size)
    X_train,X_val,y_train,y_val=train_test_split(X_trainval,y_trainval,test_size=val_frac,random_state=seed,stratify=y_trainval, shuffle=True)
    print(f" SPLIT : Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    #pos_weight pour gerer le desequilibre de classes si besoin
    n_pos=int(y_train.sum())
    n_neg=len(y_train)-n_pos
    pos_weight=n_neg/max(n_pos,1)

    #construction des dataloaders
    train_loader,val_loader,test_loader=build_dataloaders(X_train,y_train,X_val,y_val,X_test,y_test,r=r,max_dim=max_dim,batch_size=batch_size,
    )

    #config et entrainement
    cfg=TrainerConfig(
    hidden_channels=hidden_channels,
    n_layers=n_layers,
    n_epochs=n_epochs,
    lr=lr,
    dropout=dropout,
    pos_weight=pos_weight if pos_weight!=1.0 else None,
    seed=seed)
    trainer=Trainer(cfg)
    history=trainer.fit(train_loader,val_loader)
    #evaluation sur le test set
    print("\n Pipeline evaluation sur le test set")
    results=trainer.evaluate(test_loader)
    if save_path:
        trainer.save(save_path)
    return {"train_history":history,"test_results":results,"model":trainer.model,"trainer":trainer}


def main():
    parser=argparse.ArgumentParser(description="pipeline ecog -> tnn classification binaire")
    parser.add_argument("--epochs_path",required=True)
    parser.add_argument("--labels_path",required=True)
    parser.add_argument("--model",default="scnn",choices=["scnn"])
    parser.add_argument("--r",default="0.8")
    parser.add_argument("--max_dim",default=2,type=int)
    parser.add_argument("--hidden",default=32,type=int)
    parser.add_argument("--n_layers",default=2,type=int)
    parser.add_argument("--epochs",default=50,type=int)
    parser.add_argument("--batch_size",default=16,type=int)
    parser.add_argument("--lr",default=1e-3,type=float)
    parser.add_argument("--save",default=None)
    args=parser.parse_args()

    X=np.load(args.epochs_path,allow_pickle=True)
    y_raw=np.load(args.labels_path,allow_pickle=True)
    label_map={v:i for i,v in enumerate(sorted(set(y_raw)))}
    print(f"LABELS mapping : {label_map}")
    y=np.array([label_map[v] for v in y_raw])

    #si le tableau est (N,d,T) on transpose en (N,T,d)
    if X.ndim==3 and X.shape[1]<X.shape[2]:
        print(f" shape {X.shape} detectee")
        X=X.transpose(0,2,1)

    r= float(args.r)

    run_pipeline(X=X,y=y,r=r,max_dim=args.max_dim,hidden_channels=args.hidden,n_layers=args.n_layers,n_epochs=args.epochs,batch_size=args.batch_size,lr=args.lr,save_path=args.save)


if __name__=="__main__":
    main()