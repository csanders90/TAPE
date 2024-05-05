import os
import logging
import sys
import numpy as np
import time
import datetime
import pytz
import logging
import torch
import git
import subprocess
import pandas as pd
from IPython import embed
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter
from yacs.config import CfgNode as CN



def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
    import torch
    import random
    # import dgl
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path):
        return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file


def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(
            f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret

    return wrapper


def get_root_dir():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(file_dir, "..")


def get_git_repo_root_path():
    try:
        # Using git module
        git_repo = git.Repo('.', search_parent_directories=True)
        return git_repo.working_dir
    except git.InvalidGitRepositoryError:
        # Fallback to using subprocess if not a valid Git repository
        result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True)

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print("Error:", result.stderr)
            return None


# Define a function that uses the lambda function
def process_value(v):
    return (lambda x: x.tolist() if isinstance(x, torch.Tensor) else x)(v)


def append_acc_to_excel(uuid_val, metrics_acc, root, name, method):
    # if not exists save the first row
    
    csv_columns = ['Metric'] + list(k for k in metrics_acc) 

    # load old csv
    try:
        Data = pd.read_csv(root)[:-1]
    except:
        Data = pd.DataFrame(None, columns=csv_columns)
        Data.to_csv(root, index=False)
    
    # create new line 
    acc_lst = []
    
    for k, v in metrics_acc.items():
        acc_lst.append(process_value(v))
        
    # merge with old lines, 
    v_lst = [f'{name}_{uuid_val}_{method}'] + acc_lst
    new_df = pd.DataFrame([v_lst], columns=csv_columns)
    new_Data = pd.concat([Data, new_df])
    
    # best value
    highest_values = new_Data.apply(lambda column: max(column, default=None))

    # concat and save
    Best_list = ['Best'] + highest_values[1:].tolist()
    Best_df = pd.DataFrame([Best_list], columns=Data.columns)
    upt_Data = pd.concat([new_Data, Best_df])
    upt_Data.to_csv(root,index=False)

    return upt_Data


def append_mrr_to_excel(uuid_val, metrics_mrr, root, name, method):
 
    csv_columns, csv_numbers = [], []
    for i, (k, v) in enumerate(metrics_mrr.items()): 
        if i == 0:
            csv_columns = ['Metric'] + list(v.keys())
        csv_numbers.append([f'{k}_{uuid_val}_{name}_{method}'] + list(v.values()))
    
    print(csv_numbers)

    try:
        Data = pd.read_csv(root)[:-1]
    except:
        Data = pd.DataFrame(None, columns=csv_columns)
        Data.to_csv(root, index=False)

    
    new_df = pd.DataFrame(csv_numbers, columns = csv_columns)
    new_Data = pd.concat([Data, new_df])
    
    highest_values = new_Data.apply(lambda column: max(column, default=None))
    Best_list = ['Best'] + highest_values[1:].tolist()
    Best_df = pd.DataFrame([Best_list], columns=csv_columns)
    upt_Data = pd.concat([new_Data, Best_df])
    
    upt_Data.to_csv(root, index=False)

    
    return upt_Data


def config_device(cfg):
    # device 
    try:
        if cfg.device is not None:
            return cfg.device
        elif cfg.train.device is not None:
            return cfg.train.device
        elif cfg.data.device is not None:
            return cfg.data.device
    except:
        num_cuda_devices = 0
        if torch.cuda.is_available():
            # Get the number of available CUDA devices
            num_cuda_devices = torch.cuda.device_count()

        if num_cuda_devices > 0:
            # Set the first CUDA device as the active device
            torch.cuda.set_device(0)
            device = cfg.train.device
        else:
            device = 'cpu'
        
    return device


def set_cfg(file_path, args):
    with open(file_path + args.cfg_file, "r") as f:
        return CN.load_cfg(f)
    
def init_cfg_test():
    """
    Initialize a CfgNode instance to test dataloader for link prediction.

    Args:
        cfg_dict (dict): Dictionary containing configuration parameters.

    Returns:
        CN: Initialized CfgNode instance.
    """
    cfg_dict = {
        'data': {  
            'undirected': True,
            'include_negatives': True,
            'val_pct': 0.1,
            'test_pct': 0.1,
            'split_labels': True,
            'device': 'cpu'
            },
        'train':  {
                'device': 'cpu'
            }
    }
    cfg = CN(cfg_dict)
    return cfg

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            best_results = []

            for r in self.results:
                r = 100 * torch.tensor(r)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')

            r = best_result[:, 0].float()
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')

            r = best_result[:, 1].float()
            best_valid_mean = round(r.mean().item(), 2)
            best_valid_var = round(r.std().item(), 2)

            best_valid = str(best_valid_mean) +' ' + '±' +  ' ' + str(best_valid_var)
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')


            r = best_result[:, 2].float()
            best_train_mean = round(r.mean().item(), 2)
            best_train_var = round(r.std().item(), 2)
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')


            r = best_result[:, 3].float()
            best_test_mean = round(r.mean().item(), 2)
            best_test_var = round(r.std().item(), 2)
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            mean_list = [best_train_mean, best_valid_mean, best_test_mean]
            var_list = [best_train_var, best_valid_var, best_test_var]


            return best_valid, best_valid_mean, mean_list, var_list


def get_logger(name, log_dir, config_dir):
	
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


def save_emb(score_emb, save_path):

    if len(score_emb) == 6:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, x1, x2= score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        'node_emb': x1,
        'node_emb_with_valid_edges': x2

        }
        
    elif len(score_emb) == 5:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, x= score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        'node_emb': x
        }
    elif len(score_emb) == 4:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, = score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        }
   
    torch.save(state, save_path)
        

def negate_edge_index(edge_index, batch=None):
    """Negate batched sparse adjacency matrices given by edge indices.

    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.

    Implementation inspired by `torch_geometric.utils.to_dense_adj`

    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.

    Returns:
        Complementary edge index.
    """

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch,
                        dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short,
                         device=edge_index.device)

        # Remove existing edges from the full N x N adjacency matrix
        flattened_size = n * n
        adj = adj.view([flattened_size])
        _idx1 = idx1[idx0 == i]
        _idx2 = idx2[idx0 == i]
        idx = _idx1 * n + _idx2
        zero = torch.zeros(_idx1.numel(), dtype=torch.short,
                           device=edge_index.device)
        scatter(zero, idx, dim=0, out=adj, reduce='mul')

        # Convert to edge index format
        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_negative = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_negative


def flatten_dict(metrics):
    """Flatten a list of train/val/test metrics into one dict to send to wandb.

    Args:
        metrics: List of Dicts with metrics

    Returns:
        A flat dictionary with names prefixed with "train/" , "val/" , "test/"
    """
    prefixes = ['train', 'val', 'test']
    result = {}
    for i in range(len(metrics)):
        # Take the latest metrics.
        stats = metrics[i][-1]
        result.update({f"{prefixes[i]}/{k}": v for k, v in stats.items()})
    return result


def cfg_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary.

    Yacs doesn't have a default function to convert the cfg object to plain
    python dict. The following function was taken from
    https://github.com/rbgirshick/yacs/issues/19
    """
    _VALID_TYPES = {tuple, list, str, int, float, bool}

    if not isinstance(cfg_node, CN):
        if type(cfg_node) not in _VALID_TYPES:
            logging.warning(f"Key {'.'.join(key_list)} with "
                            f"value {type(cfg_node)} is not "
                            f"a valid type; valid types: {_VALID_TYPES}")
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = cfg_to_dict(v, key_list + [k])
        return cfg_dict


def make_wandb_name(cfg):
    # Format dataset name.
    dataset_name = cfg.dataset.format
    if dataset_name.startswith('OGB'):
        dataset_name = dataset_name[3:]
    if dataset_name.startswith('PyG-'):
        dataset_name = dataset_name[4:]
    if dataset_name in ['GNNBenchmarkDataset', 'TUDataset']:
        # Shorten some verbose dataset naming schemes.
        dataset_name = ""
    if cfg.dataset.name != 'none':
        dataset_name += "-" if dataset_name != "" else ""
        if cfg.dataset.name == 'LocalDegreeProfile':
            dataset_name += 'LDP'
        else:
            dataset_name += cfg.dataset.name
    # Format model name.
    model_name = cfg.model.type
    if cfg.model.type in ['gnn', 'custom_gnn']:
        model_name += f".{cfg.gnn.layer_type}"
    elif cfg.model.type == 'GPSModel':
        model_name = f"GPS.{cfg.gt.layer_type}"
    model_name += f".{cfg.name_tag}" if cfg.name_tag else ""
    # Compose wandb run name.
    name = f"{dataset_name}.{model_name}.r{cfg.run_id}"
    return name

import argparse
def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/graphsage.yaml',
                        help='The configuration file path.')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/cora/gcns/gat_sp1.yaml',
                        help='The configuration file path.')
    
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()


def set_cfg(file_path, args):
    with open(file_path + args.cfg_file, "r") as f:
        return CN.load_cfg(f)