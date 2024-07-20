import os
import sys

from torch_geometric.graphgym import params_count

# Add the project's root directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy
import time
import argparse
from data_utils.load import load_data_lp

import torch
import scipy.sparse as ssp
from torch_geometric import seed_everything

from graphgps.train.seal_train import Trainer_SEAL
from graphgps.network.heart_gnn import DGCNN
from graphgps.utility.utils import (
    set_cfg, 
    get_git_repo_root_path, 
    custom_set_run_dir, 
    set_printing, 
    run_loop_settings, 
    create_logger,
    config_device,
    save_run_results_to_csv
)

from graphgps.encoder.seal import (
    get_pos_neg_edges, 
    extract_enclosing_subgraphs, 
    k_hop_subgraph, 
    construct_pyg_graph, 
    do_edge_split,
    do_ogb_edge_split
)
from data_utils.load_data_nc import (
    load_graph_cora, 
    load_graph_pubmed, 
    load_tag_arxiv23, 
    load_graph_ogbn_arxiv
)
from data_utils.seal_loader import (
    SEALDataset, 
    SEALDynamicDataset
)

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/seal.yaml',
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
                        default=400,
                        help='data name')
    parser.add_argument('--score', dest='score', type=str, required=False, default='mlp_score',
                        help='decoder name')
    parser.add_argument('--wandb', dest='if_wandb', required=False, 
                        help='data name')
    parser.add_argument('--repeat', type=int, default=5,
                        help='The number of repeated jobs.')
    parser.add_argument('--start_seed', type=int, default=0,
                        help='The number of starting seed.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    
    return parser.parse_args()


if __name__ == "__main__":
    FILE_PATH = f'{get_git_repo_root_path()}/'

    args = parse_args()
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    cfg.data.name = args.data
    
    cfg.data.device = args.device
    cfg.model.device = args.device
    cfg.device = args.device
    cfg.train.epochs = args.epoch
    
    torch.set_num_threads(cfg.num_threads)
    batch_sizes = [cfg.train.batch_size]  # [8, 16, 32, 64]

    best_acc = 0
    best_params = {}
    loggers = create_logger(args.repeat)
    for run_id in range(args.repeat):
        seed = run_id + args.start_seed
        custom_set_run_dir(cfg, run_id)
        set_printing(cfg)
        cfg.seed = seed
        cfg.run_id = run_id
        cfg = config_device(cfg)
        seed_everything(cfg.seed)
        splits, text, data = load_data_lp[cfg.data.name](cfg.data)

        path = f'{os.path.dirname(__file__)}/seal_{cfg.data.name}'
        dataset = {}
        print_logger = set_printing(cfg)

        if cfg.train.dynamic_train == True:
            dataset['train'] = SEALDynamicDataset(
                path,
                data,
                splits,
                num_hops=cfg.model.num_hops,
                split='train',
                node_label= cfg.model.node_label,
                directed=not cfg.data.undirected,
            )
            dataset['valid'] = SEALDynamicDataset(
                path,
                data,
                splits,
                num_hops=cfg.model.num_hops,
                split='valid',
                node_label= cfg.model.node_label,
                directed=not cfg.data.undirected,
            )
            dataset['test'] = SEALDynamicDataset(
                path,
                data,
                splits,
                num_hops=cfg.model.num_hops,
                split='test',
                node_label= cfg.model.node_label,
                directed=not cfg.data.undirected,
            )
        else:
            dataset['train'] = SEALDataset(
                path,
                data,
                splits,
                num_hops=cfg.model.num_hops,
                split='train',
                node_label= cfg.model.node_label,
                directed=not cfg.data.undirected,
            )
            dataset['valid'] = SEALDataset(
                path,
                data,
                splits,
                num_hops=cfg.model.num_hops,
                split='valid',
                node_label= cfg.model.node_label,
                directed=not cfg.data.undirected,
            )
            dataset['test'] = SEALDataset(
                path,
                data,
                splits,
                num_hops=cfg.model.num_hops,
                split='test',
                node_label= cfg.model.node_label,
                directed=not cfg.data.undirected,
            )
        model = DGCNN(cfg.model.hidden_channels, cfg.model.num_layers, cfg.model.max_z, cfg.model.k,
                      dataset['train'], False, use_feature=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.base_lr)

        trainer = Trainer_SEAL(FILE_PATH,
                              cfg,
                              model,
                              optimizer,
                              data,
                              dataset,
                              run_id,
                              args.repeat,
                              loggers,
                              print_logger=print_logger,
                              batch_size=cfg.train.batch_size)

        start = time.time()
        trainer.train()
        end = time.time()
        print('Training time: ', end - start)
        save_run_results_to_csv(cfg, loggers, seed, run_id)

    print('All runs:')

    result_dict = {}
    for key in loggers:
        print(key)
        _, _, _, valid_test, _, _ = trainer.loggers[key].calc_all_stats()
        result_dict[key] = valid_test

    trainer.save_result(result_dict)

    cfg.model.params = params_count(model)
    print_logger.info(f'Num parameters: {cfg.model.params}')
    trainer.finalize()
    print_logger.info(f"Inference time: {trainer.run_result['eval_time']}")

