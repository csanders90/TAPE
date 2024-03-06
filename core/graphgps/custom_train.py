from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


def set_cfg_train(cfg):
    """Weights & Biases tracker argsuration.
    """

    # WandB group
    cfg.train = CN()

    # Use wandb or not
    cfg.train.mode = 'custom'

register_config('cfg_train', set_cfg_train)