import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graphgps.train.gsaint_train import Trainer_Saint
from graphgps.network.gsaint import GraphSAINTRandomWalkSampler, GraphSAINTNodeSampler, GraphSAINTEdgeSampler
from graphgps.network.custom_gnn import GAT, GraphSage, GCNEncoder
import time
from itertools import product
import torch
import logging
import itertools
from tqdm import tqdm
import time
from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.cmd_args import parse_args
import argparse
import wandb
from graphgps.train.opt_train import Trainer_Heart
from graphgps.network.custom_gnn import create_model
from graphgps.config import (dump_cfg, dump_run_cfg)

from data_utils.load import load_data_lp
from core.graphgps.utility.utils import set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir \
    , custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, \
        init_model_from_pretrained, create_logger
import pprint

def run_experiment(cfg, model, optimizer, splits, sampler, batch_size, walk_length, num_steps, sample_coverage):
    trainer = Trainer_Saint(FILE_PATH, 
                      cfg, 
                      model, 
                      optimizer, 
                      splits, 
                      sampler, 
                      batch_size, 
                      walk_length, 
                      num_steps, 
                      sample_coverage)
    
    start = time.time()
    trainer.train()
    end = time.time()
    
    print('Training time: ', end - start)
    results_dict = trainer.evaluate()
    trainer.save_result(results_dict)
    return results_dict

if __name__ == "__main__":
    FILE_PATH = get_git_repo_root_path() + '/'

    args = parse_args()
    cfg = set_cfg(FILE_PATH, args)
    cfg.merge_from_list(args.opts)
    
    torch.set_num_threads(cfg.num_threads)
    # Best params: {'batch_size': 64, 'walk_length': 10, 'num_steps': 30, 'sample_coverage': 100, 'accuracy': 0.82129}
    batch_sizes = [64]#[8, 16, 32, 64]
    walk_lengths = [10]#[10, 15, 20]
    num_steps = [30]#[10, 20, 30]
    sample_coverages = [100]#[50, 100, 150]

    best_acc = 0
    best_params = {}
    
    for batch_size, walk_length, num_steps, sample_coverage in product(batch_sizes, walk_lengths, num_steps, sample_coverages):
        for run_id, seed, split_index in zip(*run_loop_settings(cfg, args)): # In run_loop_settings we should send 2 parameeters
       
            # Set configurations for each run
            custom_set_run_dir(cfg, run_id)
            set_printing(cfg) # We should send cfg else we get error Attribute error: run_dir
            cfg.seed = seed
            cfg.run_id = run_id
            seed_everything(cfg.seed)
            cfg = config_device(cfg)
            dataset, data_cited, splits = data_loader[cfg.data.name](cfg)
        
            lst_args = cfg.model.type.split('_')
            if lst_args[0] == 'gsaint':
                sampler = get_loader
                cfg.model.type = lst_args[1]
            else:
                sampler = None 

            if cfg.model.type == 'GAT':
                model = LinkPredModel(GAT(cfg))
            elif cfg.model.type == 'GraphSage':
                model = LinkPredModel(GraphSage(cfg))
            elif cfg.model.type == 'GCNEncode':
                # Very strange error. Hidden_channels have string type, not int
                model = LinkPredModel(GCNEncoder(cfg))
            
            if cfg.model.type == 'gae':
                model = GAE(GCNEncoder(cfg))
            elif cfg.model.type == 'vgae':
                model = VGAE(VariationalGCNEncoder(cfg))
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
            
            # Execute experiment
            # print(f'Running experiment with batch_size={batch_size}, walk_length={walk_length}, num_steps={num_steps}, sample_coverage={sample_coverage}')
            results = run_experiment(cfg, model, optimizer, splits, sampler, batch_size, walk_length, num_steps, sample_coverage)
            
            current_acc = results['acc']
            if current_acc > best_acc:
                best_acc = current_acc
                best_params = {
                    'batch_size': batch_size,
                    'walk_length': walk_length,
                    'num_steps': num_steps,
                    'sample_coverage': sample_coverage,
                    'accuracy': best_acc
                }
            
            print('Results:', results)
        
    print("Best Parameters Found:")
    print(best_params)