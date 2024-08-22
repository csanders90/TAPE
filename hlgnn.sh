#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=normal
#SBATCH --job-name=gnn_wb

#SBATCH --nodes=1
#SBATCH --mem=501600mb
#SBATCH --output=log/TAG_Benchmark_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error
#SBATCH --gres=gpu:full:4  # Ensure you are allowed to use these many GPUs, otherwise reduce the number here
#SBATCH --chdir=/hkfs/work/workspace_haic/scratch/cc7738-TAGBench/TAPE_gerrman/TAPE/core/gcns/hl_gnn_planetoid/res_outputs

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikhelson.g@gmail.com

# Request GPU resources
source /hkfs/home/haicore/aifb/cc7738/anaconda3/etc/profile.d/conda.sh

cd /hkfs/work/workspace_haic/scratch/cc7738-TAGBench/TAPE_gerrman/TAPE/core/gcns
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12


export CUDA_VISIBLE_DEVICES=2

# running command: bash hlgnn.sh planetoid (or ogb)
if [ "$1" == "planetoid" ]; then
    # cd HL_GNN_Planetoid
    # python3 planetoid_grid.py --dataset cora --runs 1 --norm_func gcn_norm  --epochs 100 --K 20 --alpha 0.2 --init KI
    
    # python3 planetoid_tuning.py --runs 1 --lr 0.0001 --dataset pubmed --norm_func gcn_norm --mlp_num_layers 3 --hidden_channels 128 --dropout 0.1 --epochs 50 --K 20 --alpha 0.2 --init KI
    # CORA TUNED
    # python3 planetoid.py --emb_type llama --runs 10 --lr 0.0001 --dataset cora --norm_func gcn_norm --mlp_num_layers 5 --hidden_channels 8192 --dropout 0.4 --epochs 100 --K 20 --alpha 0.4 --init RWR
    #PUBMED TUNED
    python3 planetoid.py --emb_type llama --lr 0.0001 --runs 10 --dataset pubmed --norm_func gcn_norm --mlp_num_layers 5 --hidden_channels 512 --dropout 0.6 --epochs 300 --K 20 --alpha 0.2 --init KI
    # with tuning
    # python3 planetoid.py --emb_type llama --runs 10 --lr 0.0001 --dataset arxiv_2023 --norm_func gcn_norm --mlp_num_layers 3 --hidden_channels 256 --dropout 0.1 --epochs 300 --K 20 --alpha 0.2 --init RWR
    # python3 planetoid.py --emb_type bert --runs 10 --lr 0.001 --dataset arxiv_2023 --norm_func gcn_norm --mlp_num_layers 3 --hidden_channels 256 --dropout 0.1 --epochs 300 --K 20 --alpha 0.2 --init RWR
    # python3 planetoid.py --emb_type e5 --runs 10 --lr 0.001 --dataset arxiv_2023 --norm_func gcn_norm --mlp_num_layers 3 --hidden_channels 256 --dropout 0.1 --epochs 300 --K 20 --alpha 0.2 --init RWR
    # python3 planetoid.py --emb_type minilm --runs 10 --lr 0.01 --dataset arxiv_2023 --norm_func gcn_norm --mlp_num_layers 3 --hidden_channels 256 --dropout 0.1 --epochs 300 --K 20 --alpha 0.2 --init RWR

    # without fine-tuning
    # python3 planetoid.py --runs 5 --lr 0.001 --dataset cora --norm_func gcn_norm --mlp_num_layers 3 --hidden_channels 8192 --dropout 0.5 --epochs 100 --K 20 --alpha 0.2 --init RWR
    
    # with fine-tuning
    # python3 planetoid.py --emb_type llama --runs 10 --lr 0.0001 --dataset cora --norm_func gcn_norm --mlp_num_layers 4 --hidden_channels 8192 --dropout 0.6 --epochs 100 --K 20 --alpha 0.2 --init RWR
    # python3 planetoid.py --emb_type bert --runs 10 --lr 0.0001 --dataset cora --norm_func gcn_norm --mlp_num_layers 4 --hidden_channels 8192 --dropout 0.6 --epochs 100 --K 20 --alpha 0.2 --init RWR
    # python3 planetoid.py --emb_type e5 --runs 10 --lr 0.0001 --dataset cora --norm_func gcn_norm --mlp_num_layers 4 --hidden_channels 8192 --dropout 0.6 --epochs 100 --K 20 --alpha 0.2 --init RWR
    # python3 planetoid.py --emb_type minilm --runs 10 --lr 0.0001 --dataset cora --norm_func gcn_norm --mlp_num_layers 4 --hidden_channels 8192 --dropout 0.6 --epochs 100 --K 20 --alpha 0.2 --init RWR
    
    # without fine-tuning
    # python3 planetoid.py --lr 0.0001 --runs 5 --dataset pubmed --norm_func gcn_norm --mlp_num_layers 3 --hidden_channels 512 --dropout 0.6 --epochs 300 --K 20 --alpha 0.2 --init KI
    
    # with fine-tuning
    # python3 planetoid.py --emb_type llama --lr 0.0001 --runs 10 --dataset pubmed --norm_func gcn_norm --mlp_num_layers 4 --hidden_channels 512 --dropout 0.5 --epochs 300 --K 20 --alpha 0.2 --init KI
    # python3 planetoid.py --emb_type bert --lr 0.0001 --runs 10 --dataset pubmed --norm_func gcn_norm --mlp_num_layers 4 --hidden_channels 512 --dropout 0.5 --epochs 300 --K 20 --alpha 0.2 --init KI
    # python3 planetoid.py --emb_type e5 --lr 0.0001 --runs 10 --dataset pubmed --norm_func gcn_norm --mlp_num_layers 4 --hidden_channels 512 --dropout 0.5 --epochs 300 --K 20 --alpha 0.2 --init KI
    # python3 planetoid.py --emb_type minilm --lr 0.0001 --runs 10 --dataset pubmed --norm_func gcn_norm --mlp_num_layers 4 --hidden_channels 512 --dropout 0.5 --epochs 300 --K 20 --alpha 0.2 --init KI

    # python3 planetoid_tuning.py --lr 0.0001 --dataset cora --norm_func gcn_norm --mlp_num_layers 3 --hidden_channels 8192 --dropout 0.5 --epochs 100 --K 20 --alpha 0.2 --init RWR
    
    # python3 planetoid.py --dataset citeseer --runs 10 --norm_func gcn_norm --mlp_num_layers 2 --hidden_channels 8192 --dropout 0.5 --epochs 100 --K 20 --alpha 0.2 --init RWR
    
    
    # python3 planetoid_tuning.py --lr 0.0001 --dataset pubmed --runs 1 --norm_func gcn_norm --mlp_num_layers 3 --hidden_channels 512 --dropout 0.6 --epochs 100 --K 20 --alpha 0.2 --init KI
    # python3 amazon.py --dataset photo --runs 1 --norm_func col_stochastic_matrix --mlp_num_layers 3 --hidden_channels 512 --dropout 0.6 --epochs 200 --K 20 --alpha 0.2 --init RWR
    # python3 amazon.py --dataset computers --runs 10 --norm_func row_stochastic_matrix --mlp_num_layers 3 --hidden_channels 512 --dropout 0.6 --epochs 200 --K 20 --alpha 0.2 --init RWR
else

    cd OGB
    # python3 main.py --dataset ogbl-collab --runs 10 --norm_func row_stochastic_matrix --predictor DOT --use_valedges_as_input True --year 2010 --epochs 800 --eval_last_best True --dropout 0.3 --use_node_feat True
    # python3 main.py --dataset ogbl-collab --runs 10 --norm_func col_stochastic_matrix --predictor DOT --use_valedges_as_input True --year 2010 --epochs 800 --eval_last_best True --dropout 0.3 --use_node_feat True
    # python3 main.py --dataset ogbl-collab --runs 1 --norm_func gcn_norm --predictor DOT --use_valedges_as_input True --year 2010 --epochs 800 --eval_last_best True --dropout 0.3 --use_node_feat True

    # python3 main.py --dataset ogbl-ddi --runs 10 --norm_func row_stochastic_matrix --emb_hidden_channels 512 --gnn_hidden_channels 512 --mlp_hidden_channels 512 --num_neg 3 --dropout 0.3 --loss_func WeightedHingeAUC
    # python3 main.py --dataset ogbl-ddi --runs 10 --norm_func col_stochastic_matrix --emb_hidden_channels 512 --gnn_hidden_channels 512 --mlp_hidden_channels 512 --num_neg 3 --dropout 0.3 --loss_func WeightedHingeAUC
    python3 main.py --dataset ogbl-ddi --runs 1 --norm_func gcn_norm --emb_hidden_channels 512 --gnn_hidden_channels 512 --mlp_hidden_channels 512 --epochs 500 --num_neg 3 --dropout 0.3 --loss_func WeightedHingeAUC

    # python3 main.py --dataset ogbl-ppa --runs 1 --norm_func row_stochastic_matrix --emb_hidden_channels 256 --mlp_hidden_channels 512 --gnn_hidden_channels 512 --grad_clip_norm 2.0 --epochs 5 --eval_steps 1 --num_neg 3 --dropout 0.5 --use_node_feat True --alpha 0.5 --loss_func WeightedHingeAUC
    # python3 main.py --dataset ogbl-ppa --runs 1 --norm_func col_stochastic_matrix --emb_hidden_channels 256 --mlp_hidden_channels 512 --gnn_hidden_channels 512 --grad_clip_norm 2.0 --epochs 5 --eval_steps 1 --num_neg 3 --dropout 0.5 --use_node_feat True --alpha 0.5 --loss_func WeightedHingeAUC
    # python3 main.py --dataset ogbl-ppa --runs 1 --norm_func gcn_norm --emb_hidden_channels 256 --mlp_hidden_channels 512 --gnn_hidden_channels 512 --grad_clip_norm 2.0 --epochs 500 --eval_steps 1 --num_neg 3 --dropout 0.5 --use_node_feat True --alpha 0.5 --loss_func WeightedHingeAUC
    
    # python3 main.py --dataset ogbl-citation2 --runs 1 --norm_func row_stochastic_matrix --emb_hidden_channels 64 --mlp_hidden_channels 256 --gnn_hidden_channels 256 --grad_clip_norm 1.0 --epochs 20 --eval_steps 1 --num_neg 3 --dropout 0.3 --eval_metric mrr --neg_sampler local --use_node_feat True --alpha 0.6
    # python3 main.py --dataset ogbl-citation2 --runs 2 --norm_func col_stochastic_matrix --emb_hidden_channels 64 --mlp_hidden_channels 256 --gnn_hidden_channels 256 --grad_clip_norm 1.0 --epochs 3 --eval_steps 1 --num_neg 3 --dropout 0.3 --eval_metric mrr --neg_sampler local --use_node_feat True --alpha 0.6
    # python3 main.py --dataset ogbl-citation2 --runs 1 --norm_func gcn_norm --emb_hidden_channels 64 --mlp_hidden_channels 256 --gnn_hidden_channels 256 --grad_clip_norm 1.0 --epochs 100 --eval_steps 1 --num_neg 3 --dropout 0.3 --eval_metric mrr --neg_sampler local --use_node_feat True --alpha 0.6
fi
