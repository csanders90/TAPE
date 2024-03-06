import os
import argparse
from yacs.args import CfgNode as CN


def set_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #

    # Cuda device number, used for machine with multiple gpus
    cfg.device = 0
    # Whether fix the running seed to remove randomness
    cfg.seed = None
    # Number of runs with random init
    cfg.runs = 4
    cfg.embed = CN()


    # ------------------------------------------------------------------------ #
    # GNN Model options
    # ------------------------------------------------------------------------ #
    cfg.embed.model = CN()
    # GNN model name
    cfg.embed.model.name = 'node2vec'
    

                        
    # ------------------------------------------------------------------------ #



    return cfg


# Principle means that if an option is defined in a YACS args object,
# then your program should set that argsuration option using cfg.merge_from_list(opts) and not by defining,
# for example, --train-scales as a command line argument that is then used to set cfg.TRAIN.SCALES.


def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--args', default="",
                        metavar="FILE", help="Path to args file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify args options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from args file
    if os.path.isfile(args.args):
        cfg.merge_from_file(args.args)

    # Update from command line
    cfg.merge_from_list(args.opts)

    return cfg


"""
    Global variable
"""
cfg = set_cfg(CN())
