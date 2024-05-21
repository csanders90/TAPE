# modified from https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_3.ipynb
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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
from graphgps.train.opt_train import Trainer
from graphgps.network.custom_gnn import create_model
from graphgps.config import (dump_cfg, dump_run_cfg)

from data_utils.load import load_data_lp
from utils import set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir \
    , custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, \
        init_model_from_pretrained, create_logger
import pprint

FILE_PATH = f'{get_git_repo_root_path()}/'

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/gat.yaml',
                        help='The configuration file path.')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/cora/gcns/gae_sp1.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', dest='data', type=str, required=True,
                        default='pubmed',
                        help='data name')
    parser.add_argument('--batch_size', dest='bs', type=int, required=False,
                        default=2**15,
                        help='data name')
    parser.add_argument('--device', dest='device', required=True, 
                        help='device id')
    parser.add_argument('--epochs', dest='epoch', type=int, required=True,
                        default=300,
                        help='data name')
    parser.add_argument('--model', dest='model', type=str, required=True,
                        default='GAT',
                        help='data name')
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()

hyperparameter_space = {
    'GAT': {'base_lr': [0.01], 
            'batch_size': [2**10],
            'out_channels': [2**5, 2**6, 2**7, 2**8], 
            'hidden_channels':  [2**5, 2**6, 2**7, 2**8],
            'heads': [2**2, 2, 2**3], 
            'negative_slope': [0.1, 0.2], 
            'dropout': [0, 0.1], 
            'num_layers': [5, 6, 7]},
    
    'GAE': {'base_lr': [0.01, 0.015], 
            'batch_size': [2**10, 2**12, 2**13],
            'out_channels': [2**5, 2**6, 2**7, 2**8], 
            'hidden_channels': [2**5, 2**6, 2**7, 2**8]},
    
    'VGAE': {'base_lr':[0.015, 0.01],  
             'batch_size': [2**10],
             'out_channels': [2**5, 2**6, 2**7, 2**8], 
             'hidden_channels': [2**5, 2**6, 2**7, 2**8]},
    
    'GraphSage': {'base_lr': [0.015, 0.01], 
                  'batch_size': [2**10],
                  'out_channels': [2**5, 2**6, 2**7, 2**8], 
                  'hidden_channels': [2**5, 2**6, 2**7, 2**8]}
}

yaml_file = {   
             'GAT': 'core/yamls/cora/gcns/gat.yaml',
             'GAE': 'core/yamls/cora/gcns/gae.yaml',
             'VGAE': 'core/yamls/cora/gcns/vgae.yaml',
             'GraphSage': 'core/yamls/cora/gcns/graphsage.yaml'
            }


def project_main(): # sourcery skip: avoid-builtin-shadow, low-code-quality
    
    # process params
    args = parse_args()

    args.cfg_file = yaml_file[args.model]

    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    cfg.data.name = args.data
    cfg.data.device = args.device
    cfg.model.device = args.device
    cfg.device = args.device
    cfg.train.epochs = args.epoch
    cfg.model.type = args.model
    
    # save params
    custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)

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
        print_logger.info(
            f"The {cfg['data']['name']} graph with shape {splits['train']['x'].shape} is loaded on {splits['train']['x'].device},\n"
            f"Split index: {cfg['data']['split_index']} based on {data.edge_index.size(1)} samples.\n"
            f"Train: {cfg['data']['split_index'][0]}% ({2 * splits['train']['pos_edge_label'].shape[0]} samples),\n"
            f"Valid: {cfg['data']['split_index'][1]}% ({2 * splits['valid']['pos_edge_label'].shape[0]} samples),\n"
            f"Test:  {cfg['data']['split_index'][2]}% ({2 * splits['test']['pos_edge_label'].shape[0]} samples)"
        )
               
        dump_cfg(cfg)    

        hyperparameter_search = hyperparameter_space[cfg.model.type]
        print_logger.info(f"hypersearch space: {hyperparameter_search}")

        keys = hyperparameter_search.keys()
        # Generate Cartesian product of the hyperparameter values
        product = itertools.product(*hyperparameter_search.values())

        for combination in tqdm(product):
            for key, value in zip(keys, combination):
                if hasattr(cfg.model, key):
                    print_logger.info(f"Object cfg.model has attribute '{key}' with value: {getattr(cfg.model, key)}")
                    setattr(cfg.model, key, value)
                    print_logger.info(f"Object cfg.model.{key} updated to {getattr(cfg.model, key)}")
                elif hasattr(cfg.train, key):
                    print_logger.info(f"Object cfg.train has attribute '{key}' with values {getattr(cfg.train, key)}")
                    setattr(cfg.train, key, value)
                    print_logger.info(f"Object cfg.train.{key} updated to {getattr(cfg.train, key)}")
                elif hasattr(cfg.optimizer, key):    
                    print_logger.info(f"Object cfg.train has attribute '{key}' with values {getattr(cfg.optimizer, key)}")
                    setattr(cfg.train, key, value)
                    print_logger.info(f"Object cfg.train.{key} updated to {getattr(cfg.optimizer, key)}")

            print_logger.info(f"out : {cfg.model.out_channels}, hidden: {cfg.model.hidden_channels}")
            print_logger.info(f"bs : {cfg.train.batch_size}, lr: {cfg.optimizer.base_lr}")

            model = create_model(cfg)

            print_logger.info(f"{model} on {next(model.parameters()).device}" )
            # print_logger.info(cfg)
            cfg.params = params_count(model)
            print_logger.info(f'Num parameters: {cfg.params}')

            optimizer = create_optimizer(model, cfg)

            # LLM: finetuning
            if cfg.train.finetune: 
                model = init_model_from_pretrained(model, cfg.train.finetune,
                                                cfg.train.freeze_pretrained)

            hyper_id = wandb.util.generate_id()
            cfg.wandb.name_tag = f'{cfg.data.name}_run{id}_{cfg.model.type}_hyper{hyper_id}'
            custom_set_run_dir(cfg, cfg.wandb.name_tag)

            dump_run_cfg(cfg)
            print_logger.info(f"config saved into {cfg.run_dir}")
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
                print(key)
                _, _, _, test_bvalid = trainer.loggers[key].calc_run_stats(run_id, False)
                run_result[key] = test_bvalid

            for k in hyperparameter_search.keys():
                if hasattr(cfg.model, k):
                    run_result[k] = getattr(cfg.model, k)
                elif hasattr(cfg.train, k):
                    run_result[k] = getattr(cfg.train, k)
                elif hasattr(cfg.optimizer, k):
                    run_result[k] = getattr(cfg.optimizer, k)

            run_result['epochs'] = cfg.train.epochs
            run_result['train_time'] = trainer.run_result['train_time']
            run_result['test_time'] = trainer.run_result['test_time']
            run_result['params'] = cfg.model.params
            
            print_logger.info(run_result)
            to_file = f'{cfg.data.name}_{cfg.model.type}_tune_time.csv'
            trainer.save_tune(run_result, to_file)

            print_logger.info(f"train time per epoch {run_result['train_time']}")
            print_logger.info(f"test time per epoch {run_result['eval_time']}")
        

if __name__ == "__main__":
    project_main()
