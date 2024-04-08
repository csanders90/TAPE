# modified from https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_3.ipynb

import torch
import os 

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg
import torch
from embedding.tune_utils import (
    parse_args, 
    get_git_repo_root_path
)
from sklearn.metrics import *
from gae import GraphSage, GAT, LinkPredModel, GCNEncoder
from gae import set_cfg, data_loader, Trainer 


# Please don't change any parameters
args = {
    "device" : 'cuda' if torch.cuda.is_available() else 'cpu',
    "hidden_dim" : 128,
    "epochs" : 200,
}

if __name__ == "__main__":

    FILE_PATH = get_git_repo_root_path() + '/'

    args = parse_args()
    # Load args file
    
    cfg = set_cfg(FILE_PATH, args)
    cfg.merge_from_list(args.opts)

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)

    dataset, data_cited, splits = data_loader[cfg.data.name](cfg)   
    train_data, val_data, test_data = splits['train'], splits['valid'], splits['test']

    if cfg.model.type == 'GAT':
        model = LinkPredModel(GAT(cfg))
    elif cfg.model.type == 'GraphSage':
        model = LinkPredModel(GraphSage(cfg))
    elif cfg.model.type == 'GCNEncode':
        model = LinkPredModel(GCNEncoder(cfg))
        
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    trainer = Trainer(FILE_PATH,
                 cfg,
                 model, 
                 optimizer,
                 splits)
    
    trainer.train()
    results_dict = trainer.evaluate()
    
    trainer.save_result(results_dict)
    
    