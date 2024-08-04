#!/bin/bash
#SBATCH --job-name=arxiv_2023_embedding_tune
#SBATCH --output=arxiv_2023_embedding_tune.out
#SBATCH --error=arxiv_2023_embedding_tune.err
#SBATCH --time=1:00:00
#SBATCH --partition=dev_accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=152
#SBATCH --gres=gpu:4
#SBATCH --mem=501600mb

module load anaconda
source activate TAG-LP

python core/embedding_mlp/embedding_LLM_tune.py --cfg ./core/yamls/arxiv_2023/lms/llama.yaml --device cuda --epochs 10
python core/embedding_mlp/embedding_LLM_tune.py --cfg ./core/yamls/arxiv_2023/lms/minilm.yaml --device cuda --epochs 10
python core/embedding_mlp/embedding_LLM_tune.py --cfg ./core/yamls/arxiv_2023/lms/e5-large.yaml --device cuda --epochs 10

