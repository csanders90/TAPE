# modified from https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_3.ipynb
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import *
import torch
import logging
import os.path as osp 
import numpy as np
import itertools
from tqdm import tqdm
import time
from torch_geometric import seed_everything
from torch_geometric.data.makedirs import makedirs
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.cmd_args import parse_args

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from distutils.util import strtobool
import argparse

from graphgps.train.opt_train import Trainer
from graphgps.network.custom_gnn import create_model
from graphgps.config import (dump_cfg, dump_run_cfg, makedirs_rm_exist)

from data_utils.load import load_data_lp
from utils import set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir \
    , custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, \
        init_model_from_pretrained, create_logger, get_logger, StringTypeOrIntAction
import pprint

FILE_PATH = f'{get_git_repo_root_path()}/'

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/vgae.yaml',
                        help='The configuration file path.')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/cora/gcns/gae_sp1.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', dest='data', type=str, required=False,
                        default='pubmed',
                        help='data name')
    parser.add_argument('--batch_size', dest='bs', type=int, required=False,
                        default=2048,
                        help='data name')
    parser.add_argument('--device', dest='device', type=int, required=False, 
                        help='device id')
    parser.add_argument('--epoch', dest='epoch', type=int, required=False,
                        default=100,
                        help='data name')
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()

import wandb

def project_main():
    
    # process params
    args = parse_args()

    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)
    
    cfg.data.name = args.data
    cfg.data.device = args.device
    cfg.model.device = args.device
    cfg.device = args.device
    cfg.train.batch_size = args.bs
    cfg.train.epoch = args.epoch
    
    # save params
    custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)
    
    # Set Pytorch environment
    torch.set_num_threads(20)

    loggers = create_logger(args.repeat)

    for run_id, seed, split_index in zip(*run_loop_settings(cfg, args)):
        # Set configurations for each run TODO clean code here 
        id = wandb.util.generate_id()
        cfg.wandb.name_tag = f'{cfg.data.name}_run{id}_{cfg.model.type}' 
        custom_set_run_dir(cfg, cfg.wandb.name_tag)

        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        
        cfg = config_device(cfg)
        cfg.data.name = args.data

        splits, _, data = load_data_lp[cfg.data.name](cfg.data)
        cfg.model.in_channels = splits['train'].x.shape[1]

        print_logger = set_printing(cfg)
        print_logger.info(f"The {cfg['data']['name']} graph {splits['train']['x'].shape} is loaded on {splits['train']['x'].device}, \n Train: {2*splits['train']['pos_edge_label'].shape[0]} samples,\n Valid: {2*splits['train']['pos_edge_label'].shape[0]} samples,\n Test: {2*splits['test']['pos_edge_label'].shape[0]} samples")
        dump_cfg(cfg)    
        hyperparameter_search = {'lr': [0.0001, 0.001, 0.01, 0.04], 'batch_size': [1024, 2048, 4096, 8192, 16384]}
        # hyperparameter_search = {'out_channels': [16, 32, 48, 64, 80, 96, 112], 'hidden_channels': [16, 32, 48, 64, 80, 96, 112]}
        for out,  hidden in tqdm(itertools.product(*hyperparameter_search.values())):
            # cfg.model.out_channels = out
            # cfg.model.hidden_channels = hidden
            cfg.train.batch_size = hidden
            cfg.optimizer.lr = out
            
            start_time = time.time()
                
            model = create_model(cfg)
            
            logging.info(f"{model} on {next(model.parameters()).device}" )
            logging.info(cfg)
            cfg.params = params_count(model)
            logging.info(f'Num parameters: {cfg.params}')

            optimizer = create_optimizer(model, cfg)

            # LLM: finetuning
            if cfg.train.finetune: 
                model = init_model_from_pretrained(model, cfg.train.finetune,
                                                cfg.train.freeze_pretrained)
            hyper_id = wandb.util.generate_id()
            cfg.wandb.name_tag = f'{cfg.data.name}_run{id}_{cfg.model.type}_hyper{hyper_id}' 
            custom_set_run_dir(cfg, cfg.wandb.name_tag)
        
            dump_run_cfg(cfg)
            print_logger.info(f'Run {run_id} with seed {seed} and split {split_index} on device {cfg.device}')
            trainer = Trainer(FILE_PATH,
                        cfg,
                        model, 
                        None, 
                        data,
                        optimizer,
                        splits,
                        run_id, 
                        args.repeat,
                        loggers, 
                        print_logger,
                        cfg.device)

            trainer.train()

            run_result = {}
            for key in trainer.loggers.keys():
                # refer to calc_run_stats in Logger class
                _, _, _, test_bvalid = trainer.loggers[key].calc_run_stats(run_id)
                run_result.update({key: test_bvalid})
            print_logger.info(run_result)
            
            trainer.save_result(run_result)
            
            print_logger.info(f"runing time {time.time() - start_time}")
        
    # statistic for all runs


if __name__ == "__main__":
    project_main()
