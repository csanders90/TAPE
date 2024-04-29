# modified from https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_3.ipynb
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.insert(0, '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE/core')

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg, set_cfg)
from sklearn.metrics import *
import torch

from embedding.tune_utils import (
    parse_args, 
    get_git_repo_root_path
)
from gcns.example import GraphSage, GAT, LinkPredModel, GCNEncoder, GAE, VGAE, VariationalGCNEncoder
from gcns.example import set_cfg, data_loader, Trainer 

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
    
# Please don't change any parameters
args = {
    "device" : 'cuda:1' if torch.cuda.is_available() else 'cpu',
    "hidden_dim" : 128,
    "epochs" : 200,
}

if __name__ == "__main__":

    FILE_PATH = get_git_repo_root_path() + '/'

    args = parse_args()
    # Load args file
    
    cfg = set_cfg(FILE_PATH, args)
    cfg.merge_from_list(args.opts)
    custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)
    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    
    dataset, data_cited, splits = data_loader[cfg.data.name](cfg)   

    if cfg.model.type == 'GAT':
        model = LinkPredModel(GAT(cfg))
    elif cfg.model.type == 'GraphSage':
        model = LinkPredModel(GraphSage(cfg))
    elif cfg.model.type == 'GCNEncode':
        model = LinkPredModel(GCNEncoder(cfg))
    
    if cfg.model.type == 'gae':
        model = GAE(GCNEncoder(cfg))
    elif cfg.model.type == 'vgae':
        model = VGAE(VariationalGCNEncoder(cfg))
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    trainer = Trainer(FILE_PATH,
                 cfg,
                 model, 
                 optimizer,
                 splits)
    
    trainer.train()
    results_dict = trainer.evaluate()
    
    trainer.save_result(results_dict)
    
    