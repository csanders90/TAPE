
# modified from https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_3.ipynb
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pprint

# External module imports

from torch_geometric import seed_everything
from torch_geometric.data.makedirs import makedirs
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.config import (dump_cfg, 
                                             makedirs_rm_exist)
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from graphgps.train.opt_train import Trainer
from graphgps.network.custom_gnn import create_model
from data_utils.load import load_data_nc, load_data_lp
from utils import parse_args, create_optimizer, config_device, \
        init_model_from_pretrained, create_logger, set_cfg

import wandb 
from sklearn.metrics import *
from embedding.tune_utils import (

    FILE_PATH
)
import argparse

set_float = lambda result: float(result.split(' ±')[0])


def merge_cfg_from_sweep(cfg, wandb_config):
    for ind, k in wandb_config.items():
        if hasattr(cfg.model, ind):
            cfg.model.ind = k
        if hasattr(cfg.train, ind):
            cfg.train.ind = k
            
    pprint.pprint(cfg)
    pprint.pprint(wandb.config)
    return cfg



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

def run_experiment():  # sourcery skip: avoid-builtin-shadow
    
    id = wandb.util.generate_id()
    
    run = wandb.init(id=id, config=cfg_sweep, settings=wandb.Settings(_service_wait=300), save_code=True)

    wandb_config = wandb.config
    
    wandb.log(dict(wandb_config))   

    # merge model param
    cfg = merge_cfg_from_sweep(cfg_config, cfg_sweep)
    splits, _ = load_data_lp[cfg.data.name](cfg.data)
    
    torch.set_num_threads(cfg.run.num_threads)
    splits, _ = load_data_lp[cfg.data.name](cfg.data)
    cfg.model.in_channels = splits['train'].x.shape[1]
    model = create_model(cfg)

    optimizer = create_optimizer(model, cfg)
    loggers = create_logger(1)

    seed_everything(cfg.seed)
    auto_select_device()

    # LLM: finetuning
    if cfg.train.finetune: 
        model = init_model_from_pretrained(model, cfg.train.finetune,
                                            cfg.train.freeze_pretrained)
    trainer = Trainer(FILE_PATH,
                cfg,
                model, 
                optimizer,
                splits,
                0, 
                1,
                loggers)


    for epoch in range(1, cfg.train.epochs + 1):
        loss = trainer.train_func[trainer.model_name]()
        wandb.log({'loss': loss})
        
        if epoch % 100 == 0:
            results_rank = trainer.merge_result_rank()
            
            for key, result in results_rank.items():
                # result - (train, valid, test)
                loggers[key].add_result(0, result)
                # print(self.loggers[key].results)
                print(loggers[key].results)
                
    print('All runs:')
    
    result_dict = {}
    for key in loggers:
        print(key)
        _, _, _, valid_test, _, _ = trainer.loggers[key].calc_all_stats(0)
        result_dict.update({key: valid_test})

    trainer.save_result(result_dict)
    print(result_dict['Hits@100'])
    wandb.log({'Hits@100': set_float(result_dict['Hits@100'])})
    run.log_code("../", include_fn=wandb_record_files)


import torch

args = parse_args()

print(args)


cfg_sweep= 'core/yamls/cora/gcns/gae_sp1.yaml'
cfg_config = 'core/yamls/cora/gcns/gae.yaml'

cfg_sweep = set_cfg(FILE_PATH, cfg_sweep)
cfg_config = set_cfg(FILE_PATH, cfg_config)


sweep_id = wandb.sweep(sweep=cfg_sweep, project=f"{cfg_config.model.type}-sweep-{cfg_config.data.name}")

wandb.agent(sweep_id, run_experiment, count=60)
