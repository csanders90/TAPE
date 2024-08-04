#!/bin/bash
#SBATCH --job-name=pubmed_embedding_tune
#SBATCH --output=pubmed_embedding_tune.out
#SBATCH --error=pubmed_embedding_tune.err
#SBATCH --time=1:00:00
#SBATCH --partition=dev_accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=152
#SBATCH --gres=gpu:4
#SBATCH --mem=501600mb

module load anaconda
source activate TAG-LP

python core/embedding_mlp/embedding_LLM_tune.py --cfg ./core/yamls/pubmed/lms/llama.yaml --device cuda --epochs 10
python core/embedding_mlp/embedding_LLM_tune.py --cfg ./core/yamls/pubmed/lms/minilm.yaml --device cuda --epochs 10
python core/embedding_mlp/embedding_LLM_tune.py --cfg ./core/yamls/pubmed/lms/e5-large.yaml --device cuda --epochs 10

