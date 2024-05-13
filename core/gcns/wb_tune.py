
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
    record_lst = [cfg_sweep,
                  cfg_config,
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
    
    torch.set_num_threads(cfg.run.num_threads)
    splits, _, data = load_data_lp[cfg.data.name](cfg.data)
    cfg.model.in_channels = splits['train'].x.shape[1]
    model = create_model(cfg)
    
    wandb.watch(model, log="all",log_freq=10)
    optimizer = create_optimizer(model, cfg)
    loggers = create_logger(1)

    seed_everything(cfg.seed)
    cfg = config_device(cfg)

    # LLM: finetuning
    if cfg.train.finetune: 
        model = init_model_from_pretrained(model, cfg.train.finetune,
                                            cfg.train.freeze_pretrained)
    
    trainer = Trainer(FILE_PATH,
                cfg,
                model, 
                None, 
                data,
                optimizer,
                splits,
                0, 
                args.repeat,
                loggers)

    best_auc, best_hits, best_hit100 = 0, 0, 0
    
    results_rank = {}
    for epoch in range(1, cfg.train.epochs + 1):
        loss = trainer.train_func[cfg.model.type]()
        
        if epoch % 100 == 0:
            results_rank = trainer.merge_result_rank()
            # print(results_rank)
            
            for key, result in results_rank.items():   
                trainer.loggers[key].add_result(0, result)
                
            print(f'Epoch: {epoch:03d}, Loss_train: {loss:.4f}, AUC: {results_rank["AUC"][0]:.4f}, AP: {results_rank["AP"][0]:.4f}, MRR: {results_rank["MRR"][0]:.4f}, Hit@10 {results_rank["Hits@100"][0]:.4f}')
            print(f'Epoch: {epoch:03d}, Loss_train: {loss:.4f}, AUC: {results_rank["AUC"][1]:.4f}, AP: {results_rank["AP"][1]:.4f}, MRR: {results_rank["MRR"][1]:.4f}, Hit@10 {results_rank["Hits@100"][1]:.4f}')               
            print(f'Epoch: {epoch:03d}, Loss_train: {loss:.4f}, AUC: {results_rank["AUC"][2]:.4f}, AP: {results_rank["AP"][2]:.4f}, MRR: {results_rank["MRR"][2]:.4f}, Hit@10 {results_rank["Hits@100"][2]:.4f}')               

            if results_rank["AUC"][1] > best_auc:
                best_auc = results_rank["AUC"][1]
            elif results_rank['Hits@100'][1] > best_hit100:
                best_hits = results_rank['Hits@100'][1]
                
        for key, result in results_rank.items():
            trainer.loggers[key].add_result(0, result)
            if epoch % 500 == 0:
                for key, result in results_rank.items():
                    print(key)
                    train_hits, valid_hits, test_hits = result
                    print(
                        f'Run: {0 + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train: {100 * train_hits:.2f}%, '
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
                print('---')
                
    result_dict = {}
    for key in loggers:
        print(key)
        _, _, _, valid_test, _, _ = trainer.loggers[key].calc_all_stats()
        result_dict.update({key: valid_test})
        
    trainer.save_result(result_dict)
    wandb.log({'Hits@100': set_float(result_dict['Hits@100'])})
    
    wandb.log({'best hits100': best_hits})
    wandb.log({'best auc': best_auc})
    return  set_float(result_dict['Hits@100'])

import torch

args = parse_args()

print(args)

# cfg_sweep= 'core/yamls/cora/gcns/gae_sp1.yaml'
# cfg_config = 'core/yamls/cora/gcns/gae.yaml'

cfg_sweep= 'core/yamls/pubmed/gcns/gae_sp1.yaml'
cfg_config = 'core/yamls/pubmed/gcns/gae.yaml'

cfg_sweep = set_cfg(FILE_PATH, cfg_sweep)
cfg_config = set_cfg(FILE_PATH, cfg_config)


sweep_id = wandb.sweep(sweep=cfg_sweep, project=f"{cfg_config.model.type}-sweep-{cfg_config.data.name}")

wandb.agent(sweep_id, run_experiment, count=60)


# TODO multirun weight baises trainer 