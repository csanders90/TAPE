# modified from https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_3.ipynb
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import *
import torch
import torch.optim as optim
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
from utils import set_cfg, parse_args, get_git_repo_root_path
from graphgps.finetuning import get_final_pretrained_ckpt


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run.multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run.multiple_splits)
        seeds = [cfg.run.seed] * num_iterations
        split_indices = cfg.run.multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def set_printing():
    """
    Set up printing options

    """
    logging.root.handlers = []
    logging_cfg = {'level': logging.INFO, 'format': '%(message)s'}
    makedirs(cfg.run_dir)
    h_file = logging.FileHandler(f'{cfg.run_dir}/logging.log')
    h_stdout = logging.StreamHandler(sys.stdout)
    if cfg.print == 'file':
        logging_cfg['handlers'] = [h_file]
    elif cfg.print == 'stdout':
        logging_cfg['handlers'] = [h_stdout]
    elif cfg.print == 'both':
        logging_cfg['handlers'] = [h_file, h_stdout]
    else:
        raise ValueError('Print option not supported')
    logging.basicConfig(**logging_cfg)


def create_optimizer(model, optimizer_config):
    # sourcery skip: list-comprehension
    r"""
    Create optimizer for the model

    Args:
        params: PyTorch model parameters

    Returns: PyTorch optimizer

    """
    params = []

    params.extend(
        param for _, param in model.named_parameters() if param.requires_grad
    )
    optimizer = optimizer_config.optimizer
    if optimizer.type == 'adam':
        optimizer = optim.Adam(params, lr=optimizer.base_lr)
    elif optimizer.type == 'sgd':
        optimizer = optim.SGD(params, lr=optimizer.base_lr)
    else:
        raise ValueError(f'Optimizer {optimizer_config.optimizer} not supported')

    return optimizer


def create_scheduler(optimizer, scheduler_config):
    r"""
    Create learning rate scheduler for the optimizer

    Args:
        optimizer: PyTorch optimizer

    Returns: PyTorch scheduler

    """

    # Try to load customized scheduler
    if scheduler_config.scheduler == 'none':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_config.max_epoch + 1)
    elif scheduler_config.scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=scheduler_config.steps,
            gamma=scheduler_config.lr_decay)
    elif scheduler_config.scheduler == 'cos':
        scheduler = \
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.max_epoch)
    else:
        raise ValueError(f'Scheduler {scheduler_config.scheduler} not supported')
    return scheduler


def init_model_from_pretrained(model, pretrained_dir, freeze_pretrained=False):
    """ Copy model parameters from pretrained model except the prediction head.

    Args:
        model: Initialized model with random weights.
        pretrained_dir: Root directory of saved pretrained model.
        freeze_pretrained: If True, do not finetune the loaded pretrained
            parameters, train the prediction head only. If False, train all.

    Returns:
        Updated pytorch model object.
    """
    ckpt_file = get_final_pretrained_ckpt(osp.join(pretrained_dir, '0', 'ckpt'))
    logging.info(f"[*] Loading from pretrained model: {ckpt_file}")

    ckpt = torch.load(ckpt_file)
    pretrained_dict = ckpt['model_state']
    model_dict = model.state_dict()

    # Filter out prediction head parameter keys.
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if not k.startswith('post_mp')}
    # Overwrite entries in the existing state dict.
    model_dict.update(pretrained_dict)
    # Load the new state dict.
    model.load_state_dict(model_dict)

    if freeze_pretrained:
        for key, param in model.named_parameters():
            if not key.startswith('post_mp'):
                param.requires_grad = False
    return model



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

    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()

        splits, text = load_data_lp[cfg.data.name](cfg.data)
        in_channels, out_channels = splits['train'].x.shape[1], cfg.model.out_channels
        cfg.model.in_channels = in_channels
        cfg.out_channels = out_channels
        
        model = create_model(cfg)
        
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info(f'Num parameters: {cfg.params}')

        # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer = create_optimizer(model, cfg)

        # LLM: finetuning
        if cfg.train.finetune: 
            model = init_model_from_pretrained(model, cfg.train.finetune,
                                               cfg.train.freeze_pretrained)

        trainer = Trainer(FILE_PATH,
                    cfg,
                    model, 
                    optimizer,
                    splits)

        trainer.train()
        results_dict = trainer._evaluate()

        trainer.save_result(results_dict)
        
    