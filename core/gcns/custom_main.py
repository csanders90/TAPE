# modified from https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_3.ipynb
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import *
import torch
import logging
import os.path as osp 

from torch_geometric import seed_everything
from torch_geometric.data.makedirs import makedirs
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg, 
                                             makedirs_rm_exist, set_cfg)
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from graphgps.train.opt_train import Trainer 
from graphgps.network.custom_gnn import create_model
from data_utils.load import load_data_nc, load_data_lp
from utils import set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir \
    , custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, \
        init_model_from_pretrained, create_logger


if __name__ == "__main__":

    FILE_PATH = f'{get_git_repo_root_path()}/'

    args = parse_args()
    # Load args file

    cfg = set_cfg(FILE_PATH, args)
    cfg.merge_from_list(args.opts)
    custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)
    dump_cfg(cfg)

    # Set Pytorch environment
    torch.set_num_threads(cfg.run.num_threads)

    loggers = create_logger(args)
    
    for run_id, seed, split_index in zip(*run_loop_settings(cfg, args)):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
    
        set_printing(cfg)
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()

        splits, text = load_data_lp[cfg.data.name](cfg.data)
        cfg.model.in_channels = splits['train'].x.shape[1]
        model = create_model(cfg)
        
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info(f'Num parameters: {cfg.params}')

        optimizer = create_optimizer(model, cfg)

        # LLM: finetuning
        if cfg.train.finetune: 
            model = init_model_from_pretrained(model, cfg.train.finetune,
                                               cfg.train.freeze_pretrained)
        
        trainer = Trainer(FILE_PATH,
                    cfg,
                    model, 
                    optimizer,
                    splits,
                    run_id, 
                    args.repeat,
                    loggers)

        best_auc, best_hits = trainer.train()
        
        for key in loggers:
            print(key)
            trainer.loggers[key].print_statistics(run_id)
            
        print(trainer.results_rank)

    best_auc_metric, result_all_run = trainer.result_statistic()
    
    print(f"best_auc_metric: {best_auc_metric}, result_all_run: {result_all_run}")