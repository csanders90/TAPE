# modified from https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_3.ipynb
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import *
import torch
import logging
import os.path as osp 

from torch_geometric import seed_everything
from torch_geometric.data.makedirs import makedirs
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (dump_cfg, 
                                             makedirs_rm_exist)
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from graphgps.train.opt_train import Trainer
from graphgps.network.custom_gnn import create_model
from data_utils.load import load_data_nc, load_data_lp
from utils import set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir \
    , custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, \
        init_model_from_pretrained, create_logger
from embeddings.embedding_generation import *

print("modules loaded")

if __name__ == "__main__":

    FILE_PATH = f'{get_git_repo_root_path()}/'

    args = parse_args()
    # Load args file
    
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)
    custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)
    dump_cfg(cfg)

    # Set Pytorch environment
    torch.set_num_threads(cfg.run.num_threads)

    loggers = create_logger(args.repeat)
    
    splits, text = load_data_lp[cfg.data.name](cfg.data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # LLM: finetuning
    sentence_embedding = True
    if sentence_embedding == True:
        model_name = "sentence-transformers/all-mpnet-base-v2"
        # model_name = "BAAI/bge-large-en-v1.5"
        # model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = create_sentence_embeddings(model_name, text, device, with_preprocessing=False)
        
    llm_embedding = False
    if llm_embedding == True:
        # model_name = "meta-llama/Llama-2-7b-hf"
        model_name = "meta-llama/Meta-Llama-3-8B"
        embeddings = create_llm_embeddings_from_pretrained(model_name, text, device, with_preprocessing=False, batch_size=4)
        
    openai_embedding = False
    if openai_embedding == True:
        model_name = "text-embedding-3-small"
        embeddings = create_openai_embeddings(model_name, text)
        print(embeddings)
        print(embeddings.device)

    tfidf_embedding = False
    if tfidf_embedding == True:
        embeddings = create_tfidf_embeddings(text, with_preprocessing=True)

    for split in splits:
        splits[split].x = embeddings

    for run_id, seed, split_index in zip(*run_loop_settings(cfg, args)):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)

        set_printing(cfg)
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()

        cfg.model.in_channels = splits['train'].x.shape[1]
        model = create_model(cfg)

        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info(f'Num parameters: {cfg.params}')

        optimizer = create_optimizer(model, cfg)


        trainer = Trainer(FILE_PATH,
                    cfg,
                    model, 
                    optimizer,
                    splits,
                    run_id, 
                    args.repeat,
                    loggers)

        trainer.train()

    # statistic for all runs
    print('All runs:')
    
    result_dict = {}
    for key in loggers:
        print(key)
        _, _, _, valid_test, _, _ = trainer.loggers[key].calc_all_stats()
        result_dict.update({key: valid_test})

    trainer.save_result(model_name, result_dict)
