import copy
import os, sys
from sentence_transformers import SentenceTransformer
from torch import Tensor, nn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.device import auto_select_device
from graphgps.utility.utils import set_cfg, get_git_repo_root_path, custom_set_run_dir, set_printing, run_loop_settings, \
    create_optimizer, config_device, \
    create_logger
from torch_geometric.graphgym.utils.comp_budget import params_count
from data_utils.load import load_data_lp, load_graph_lp
from graphgps.train.embedding_LLM_train import Trainer_embedding_LLM
from graphgps.utility.utils import save_run_results_to_csv

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/ncnc.yaml',
                        help='The configuration file path.')
    parser.add_argument('--repeat', type=int, default=5,
                        help='The number of repeated jobs.')
    parser.add_argument('--start_seed', type=int, default=0,
                        help='The number of starting seed.')
    parser.add_argument('--device', dest='device', required=True,
                        help='device id')
    parser.add_argument('--epochs', dest='epoch', type=int, required=True,
                        default=1000,
                        help='data name')
    parser.add_argument('--wandb', dest='wandb', required=False,
                        help='data name')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    return parser.parse_args()

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

'''class LinkPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(input_dim * 2, hidden_dim))  # Concatenate xi and xj
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
        self.lins.append(nn.Linear(hidden_dim, output_dim))

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers


        self.output_layer = nn.Linear(hidden_dim, 1)  # Output layer for binary classification

        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, xi, xj):
        x = torch.cat((xi, xj), dim=1)  # Concatenate xi and xj
        x = self.dropout(x)
        h = x
        for i in range(self.num_layers - 1):

            h = F.relu(self.batch_norms[i](self.lins[i](h)))
            h = self.dropout(h)

        h = self.lins[-1](h)
        output = torch.sigmoid(h)
        return output
'''
if __name__ == '__main__':
    FILE_PATH = f'{get_git_repo_root_path()}/'

    args = parse_args()
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    cfg.data.device = args.device
    cfg.model.device = args.device
    cfg.device = args.device
    cfg.train.epochs = args.epoch

    torch.set_num_threads(cfg.num_threads)
    best_acc = 0
    best_params = {}
    loggers = create_logger(args.repeat)
    cfg.device = args.device
    splits, text, data = load_data_lp[cfg.data.name](cfg.data)
    if cfg.embedder.type == 'minilm':
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        node_features = model.encode(text)
    elif cfg.embedder.type == 'e5-large':
        model = SentenceTransformer('intfloat/e5-large-v2')
        node_features = model.encode(text, normalize_embeddings=True)
    '''elif cfg.embedder.type == 'text-embedding-ada-002':
        tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/text-embedding-ada-002')
        for i in range(len(text)):
            node_features.append(tokenizer.encode(text[i]))
        node_features = torch.tensor(node_features)'''
    node_features = torch.tensor(node_features)
    print(node_features.shape)

    for run_id in range(args.repeat):
        seed = run_id + args.start_seed
        custom_set_run_dir(cfg, run_id)
        set_printing(cfg)
        print_logger = set_printing(cfg)
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        cfg = config_device(cfg)

        model = LinkPredictor(node_features.shape[1], cfg.model.hidden_channels, 1, cfg.model.num_layers, cfg.model.dropout)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.optimizer.base_lr, weight_decay=cfg.optimizer.weight_decay)
        trainer = Trainer_embedding_LLM(FILE_PATH,
                                             cfg,
                                             model,
                                             optimizer,
                                             node_features,
                                             splits,
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


