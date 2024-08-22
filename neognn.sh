#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=normal
#SBATCH --job-name=gnn_wb

#SBATCH --nodes=1
#SBATCH --mem=501600mb
#SBATCH --output=log/TAG_Benchmark_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error
#SBATCH --gres=gpu:full:4  # Ensure you are allowed to use these many GPUs, otherwise reduce the number here
#SBATCH --chdir=/hkfs/work/workspace_haic/scratch/cc7738-TAGBench/TAPE_gerrman/TAPE/core/gcns/res_outputs_neognn

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikhelson.g@gmail.com

# Request GPU resources
source /hkfs/home/haicore/aifb/cc7738/anaconda3/etc/profile.d/conda.sh

cd /hkfs/work/workspace_haic/scratch/cc7738-TAGBench/TAPE_gerrman/TAPE/core/gcns
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12


export CUDA_VISIBLE_DEVICES=0
# python3 neognn_tune.py --data arxiv_2023 --device cuda --emb_name llama --epochs 30 --repeat 1 --cfg ./core/yamls/arxiv_2023/gcns/neognn.yaml
# python3 neognn_tune.py --data pubmed --device cuda --emb_name llama --epochs 30 --repeat 1 --cfg ./core/yamls/pubmed/gcns/neognn.yaml
# python3 neognn_main.py --emb_name llama --epochs 500 --repeat 5 --cfg ./core/yamls/cora/gcns/neognn_llama.yaml --data cora --device cuda 

python3 neognn_main.py --emb_name llama --epochs 500 --repeat 5 --cfg ./core/yamls/arxiv_2023/gcns/neognn_llama.yaml --data arxiv_2023 --device cuda 
# python3 neognn_main.py --emb_name llama --epochs 500 --repeat 3 --cfg ./core/yamls/pubmed/gcns/neognn_llama.yaml --data pubmed --device cuda 
# python3 neognn_main.py --emb_name bert --epochs 500 --repeat 5 --cfg ./core/yamls/arxiv_2023/gcns/neognn.yaml --data arxiv_2023 --device cuda
# python3 neognn_main.py --emb_name e5 --epochs 500 --repeat 5 --cfg ./core/yamls/arxiv_2023/gcns/neognn.yaml --data arxiv_2023 --device cuda
# python3 neognn_main.py --emb_name minilm --epochs 500 --repeat 4 --cfg ./core/yamls/arxiv_2023/gcns/neognn.yaml --data arxiv_2023 --device cuda

# Wait for all background processes to complete
wait
