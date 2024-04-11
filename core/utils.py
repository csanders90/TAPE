import os
import numpy as np
import time
import datetime
import pytz
import random

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


# * ============================= Time Related =============================


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

import git
import subprocess

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

import pandas as pd
import torch 
import csv
import uuid
from IPython import embed

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
        if cfg.data.device is not None:
            return cfg.data.device
        elif cfg.train.device is not None:
            return cfg.train.device
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



def init_seed(seed=2020):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    
    
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


import logging, sys
def get_logger(name, log_dir, config_dir):
	
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger