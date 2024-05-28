import copy
import os, sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse 
import time
import torch
from graphgps.train.seal_train import Trainer_SEAL
from graphgps.network.heart_gnn import DGCNN

from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.device import auto_select_device
from graphgps.utility.utils import set_cfg, get_git_repo_root_path, custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, \
    create_logger

from graphgps.encoder.seal import get_pos_neg_edges, extract_enclosing_subgraphs, k_hop_subgraph, construct_pyg_graph, do_edge_split

from torch_geometric.data import InMemoryDataset, Dataset
from data_utils.load_data_nc import load_graph_cora, load_graph_pubmed, load_tag_arxiv23, load_graph_ogbn_arxiv
import scipy.sparse as ssp

from data_utils.load import load_data_lp

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/seal.yaml',
                        help='The configuration file path.')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/cora/gcns/gae_sp1.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', dest='data', type=str, required=False,
                        default='cora',
                        help='data name')
        
    parser.add_argument('--repeat', type=int, default=2,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()

class SEALDataset(InMemoryDataset):
    def __init__(self, root, data, splits, num_hops, percent=100, split='train',node_label='drnl', ratio_per_hop=1.0,
                 max_nodes_per_hop=None, directed=False):
        self.data = data
        self.split_edge = splits
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        super(SEALDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_file_names(self):
        if self.percent == 100:
            name = 'SEAL_{}_data'.format(self.split)
        else:
            name = 'SEAL_{}_data_{}'.format(self.split, self.percent)
        name += '.pt'
        return [name]

    def process(self):
        pos_edge, neg_edge = self.split_edge[self.split]['pos_edge_label_index'], self.split_edge[self.split]['neg_edge_label_index']
        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)

        edge_weight_cpu = edge_weight.cpu()
        edge_index_cpu = self.data.edge_index.cpu()

        A = ssp.csr_matrix(
            (edge_weight_cpu, (edge_index_cpu[0], edge_index_cpu[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )

        A_csc = A.tocsc() if self.directed else None
        # Extract enclosing subgraphs for pos and neg edges
        pos_list = extract_enclosing_subgraphs(
            pos_edge, A, self.data.x, 1, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)
        neg_list = extract_enclosing_subgraphs(
            neg_edge, A, self.data.x, 0, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)

        torch.save(self.collate(pos_list + neg_list), self.processed_paths[0])
        del pos_list, neg_list


class SEALDynamicDataset(Dataset):
    def __init__(self, root, data, splits, num_hops, percent=100, split='train',
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0,
                 max_nodes_per_hop=None, directed=False, **kwargs):
        self.data = data
        self.split_edge = splits
        self.num_hops = num_hops
        self.percent = percent
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        super(SEALDynamicDataset, self).__init__(root)

        pos_edge, neg_edge = get_pos_neg_edges(split, self.split_edge,
                                               self.data.edge_index,
                                               self.data.num_nodes,
                                               self.percent)
        self.links = torch.cat([pos_edge, neg_edge], 1).t().tolist()
        self.labels = [1] * pos_edge.size(1) + [0] * neg_edge.size(1)


        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        edge_weight_cpu = edge_weight.cpu()
        edge_index_cpu = self.data.edge_index.cpu()

        self.A = ssp.csr_matrix(
            (edge_weight_cpu, (edge_index_cpu[0], edge_index_cpu[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
        self.A_csc = self.A.tocsc() if self.directed else None

    def __len__(self):
        return len(self.links)

    def len(self):
        return self.__len__()

    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]
        tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, self.ratio_per_hop,
                             self.max_nodes_per_hop, node_features=self.data.x,
                             y=y, directed=self.directed, A_csc=self.A_csc)
        data = construct_pyg_graph(*tmp, self.node_label)

        return data


if __name__ == "__main__":
    FILE_PATH = f'{get_git_repo_root_path()}/'

    args = parse_args()
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    torch.set_num_threads(cfg.num_threads)
    batch_sizes = [cfg.train.batch_size]


    best_acc = 0
    best_params = {}
    loggers = create_logger(args.repeat)
    for batch_size in batch_sizes:
        for run_id, seed, split_index in zip(
                *run_loop_settings(cfg, args)):
            custom_set_run_dir(cfg, run_id)
            set_printing(cfg)
            cfg.seed = seed
            cfg.run_id = run_id
            seed_everything(cfg.seed)
            splits, text, data = load_data_lp[cfg.data.name](cfg)
            data.edge_index = splits['train']['pos_edge_label_index']

            path = f'{os.path.dirname(__file__)}/seal_{cfg.data.name}'
            dataset = {}

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

            # Execute experiment
            trainer = Trainer_SEAL(FILE_PATH,
                                   cfg,
                                   model,
                                   optimizer,
                                   dataset,
                                   run_id,
                                   args.repeat,
                                   loggers,
                                   print_logger = None,
                                   batch_size=batch_size)

            start = time.time()
            trainer.train()
            end = time.time()
            print('Training time: ', end - start)

        print('All runs:')

        result_dict = {}
        for key in loggers:
            print(key)
            _, _, _, valid_test, _, _ = trainer.loggers[key].calc_all_stats()
            result_dict[key] = valid_test

        trainer.save_result(result_dict)
