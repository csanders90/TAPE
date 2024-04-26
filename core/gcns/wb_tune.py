
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import pprint
import numpy as np

# External module imports
import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.nn import GCNConv
from heuristic.eval import get_metric_score
from data_utils.load_cora_lp import get_cora_casestudy 
from data_utils.load_pubmed_lp import get_pubmed_casestudy
from data_utils.load_arxiv2023_lp import get_raw_text_arxiv_2023
from textfeat.mlp_dot_product import data_loader, FILE_PATH, set_cfg

from utils import config_device
from IPython import embed
import wandb 
from sklearn.metrics import *
from embedding.tune_utils import (
    set_cfg,
    parse_args,
    load_sweep_config, 
    initialize_config, 
    param_tune_acc_mrr,
    process_edge_index,
    FILE_PATH
)
from gnn_models import GraphSage, GAT, LinkPredModel, GCNEncoder, GAE, VGAE, VariationalGCNEncoder, Trainer
import argparse

def merge_cfg_from_sweep(cfg, wandb_config):
    for ind, k in wandb_config.items():
        if hasattr(cfg.model, ind):
            cfg.model.ind = k
        if hasattr(cfg.train, ind):
            cfg.train.ind = k
    
    pprint.pprint(cfg)
    pprint.pprint(wandb.config)
    return cfg

def train_and_evaluate(id, splits, cfg, wandb_config):
    
    cfg = merge_cfg_from_sweep(cfg, wandb_config)
    
    if cfg.model.type == 'GAT':
        model = LinkPredModel(GAT(cfg))
    elif cfg.model.type == 'GraphSage':
        model = LinkPredModel(GraphSage(cfg))
    elif cfg.model.type == 'GCNEncode':
        model = LinkPredModel(GCNEncoder(cfg))
    
    if cfg.model.type == 'gae':
        model = GAE(GCNEncoder(cfg))
    elif cfg.model.type == 'vgae':
        model = VGAE(VariationalGCNEncoder(cfg))
        
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb_config.lr)

    trainer = Trainer(FILE_PATH,
                    cfg,
                    model, 
                    optimizer,
                    splits)
    
    best_auc, best_hits, results_dict = trainer.train()

    trainer.save_result(results_dict)

    root = FILE_PATH + 'results'
    mrr_file = root + f'/{cfg.data.name}_mrr.csv'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    param_tune_acc_mrr(id, results_dict, mrr_file, cfg.data.name, cfg.model.type)

    return best_auc, best_hits


args = parse_args()

print(args)

SWEEP_FILE_PATH = FILE_PATH + args.sweep_file
sweep_config = load_sweep_config(SWEEP_FILE_PATH)

cfg = initialize_config(FILE_PATH, args)

_, _, splits = data_loader[cfg.data.name](cfg)


def wandb_record_files(path):
    record_or_not = False
    record_lst = [args.sweep_file, 
                  args.cfg_file, 
                  'core/gcns/wb_tune.py'
                  ]
    
    for recorded in record_lst:
        if recorded in path:
            record_or_not = True
            break
    return record_or_not

def run_experiment(config=None):
    id = wandb.util.generate_id()
    run = wandb.init(id=id, config=config, settings=wandb.Settings(_service_wait=300), save_code=True)

    wandb_config = wandb.config
    
    wandb.log(dict(wandb_config))

    best_auc, best_hits = train_and_evaluate(id, splits, cfg, wandb_config)
    run.log({"score": best_auc})
    run.log_code("../", include_fn=wandb_record_files)


sweep_id = wandb.sweep(sweep=sweep_config, project=f"{cfg.model.type}-sweep-{cfg.data.name}")

wandb.agent(sweep_id, run_experiment, count=60)
