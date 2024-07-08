import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from torch_geometric import seed_everything
# Assuming other necessary imports from your script
from graphgps.utility.utils import (
    set_cfg, get_git_repo_root_path, Logger, custom_set_out_dir,
    custom_set_run_dir, set_printing, run_loop_settings, create_optimizer,
    config_device, init_model_from_pretrained, create_logger, use_pretrained_llm_embeddings
)
import scipy.sparse as ssp
from torch_geometric.graphgym.config import dump_cfg, makedirs_rm_exist
from data_utils.load import load_data_nc, load_data_lp
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from typing import List 
import scipy
import argparse
import time 


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/gae.yaml',
                        help='The configuration file path.')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/cora/gcns/gae_sp1.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', dest='data', type=str, required=False,
                        default='cora',
                        help='data name')
    parser.add_argument('--seed', dest='seed', type=int, required=False,
                        default=2,
                        help='random seed')        
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()


def get_word2vec_embeddings(model, text):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def get_embeddings(text):
    if embedding_model_type == "tfidf":
        return embedding_model.encode(text)
    elif embedding_model_type == "word2vec":
        return get_word2vec_embeddings(embedding_model, text)
    else:
        return embedding_model.encode(text)

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
    
def process_edges(pos_edge_index, neg_edge_index, text):
    dataset = []
    labels = []
    # Process positive edges
    for i in tqdm(range(pos_edge_index.shape[1])):
        node1 = pos_edge_index[0, i].item()
        node2 = pos_edge_index[1, i].item()
        text1 = text[node1]
        text2 = text[node2]
        embedding_text1 = get_embeddings(text1)
        embedding_text2 = get_embeddings(text2)
        combined_embedding = np.concatenate((embedding_text1, embedding_text2))
        dataset.append(combined_embedding)
        labels.append(1)

    # Process negative edges
    for i in tqdm(range(neg_edge_index.shape[1])):
        node1 = neg_edge_index[0, i].item()
        node2 = neg_edge_index[1, i].item()
        text1 = text[node1]
        text2 = text[node2]
        embedding_text1 = get_embeddings(text1)
        embedding_text2 = get_embeddings(text2)
        combined_embedding = np.concatenate((embedding_text1, embedding_text2))
        dataset.append(combined_embedding)
        labels.append(0)
    
    return dataset, labels

def process_texts(pos_edge_index, neg_edge_index, text):
    dataset = []
    labels = []
    # Process positive edges
    for i in tqdm(range(pos_edge_index.shape[1])):
        node1 = pos_edge_index[0, i].item()
        node2 = pos_edge_index[1, i].item()
        text1 = text[node1]
        text2 = text[node2]
        combined_text = text1 + " " + text2
        dataset.append(combined_text)
        labels.append(1)

    # Process negative edges
    for i in tqdm(range(neg_edge_index.shape[1])):
        node1 = neg_edge_index[0, i].item()
        node2 = neg_edge_index[1, i].item()
        text1 = text[node1]
        text2 = text[node2]
        combined_text = text1 + " " + text2
        dataset.append(combined_text)
        labels.append(0)
    
    return dataset, labels

def save_dataset(embedding_model_name, cfg, args):
    # create dataset with 3 seeds
    
    for run_id, seed, split_index in zip(*run_loop_settings(cfg, args)):
        print(f'run id : {run_id}, seed: {seed}, split_index: {split_index}')
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        splits, text, _ = load_data_lp[cfg.data.name](cfg.data)

        if embedding_model_name == "tfidf":
            train_dataset, train_labels = process_texts(
                splits['train'].pos_edge_label_index, 
                splits['train'].neg_edge_label_index, 
                text
            )
            val_dataset, val_labels = process_texts(
                splits['valid'].pos_edge_label_index, 
                splits['valid'].neg_edge_label_index, 
                text
            )
            test_dataset, test_labels = process_texts(
                splits['test'].pos_edge_label_index, 
                splits['test'].neg_edge_label_index, 
                text
            )
            
            vectorizer = TfidfVectorizer()
            
            os.makedirs(f'./generated_dataset/{cfg.data.name}/', exist_ok=True)
            start_time = time.time()
            train_dataset = vectorizer.fit_transform(train_dataset)
            print(f'fit_transform: {time.time() - start_time:.2f} seconds')

            # del train_dataset
            start_time = time.time()
            val_dataset = vectorizer.transform(val_dataset)
            print(f'fit_transform: {time.time() - start_time:.2f} seconds')
            
            start_time = time.time()
            test_dataset = vectorizer.transform(test_dataset)
            print(f'fit_transform: {time.time() - start_time:.2f} seconds')
            

        elif embedding_model_name == "word2vec":
            sentences = [text[i].split() for i in range(len(text))]
            word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
            train_dataset, train_labels = process_edges(
                splits['train'].pos_edge_label_index, 
                splits['train'].neg_edge_label_index, 
                text, 
                word2vec_model, 
                "word2vec"
            )
            val_dataset, val_labels = process_edges(
                splits['valid'].pos_edge_label_index, 
                splits['valid'].neg_edge_label_index, 
                text, 
                word2vec_model, 
                "word2vec"
            )
            test_dataset, test_labels = process_edges(
                splits['test'].pos_edge_label_index, 
                splits['test'].neg_edge_label_index, 
                text, 
                word2vec_model, 
                "word2vec"
            )
        else:
            embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            train_dataset, train_labels = process_edges(
                splits['train'].pos_edge_label_index, 
                splits['train'].neg_edge_label_index, 
                text, 
                embedding_model, 
                "mpnet"
            )
            val_dataset, val_labels = process_edges(
                splits['valid'].pos_edge_label_index, 
                splits['valid'].neg_edge_label_index, 
                text, 
                embedding_model, 
                "mpnet"
            )
            test_dataset, test_labels = process_edges(
                splits['test'].pos_edge_label_index, 
                splits['test'].neg_edge_label_index, 
                text, 
                embedding_model, 
                "mpnet"
            )
            
        return train_dataset, train_labels, val_dataset, val_labels, test_dataset, test_labels
        # ssp.save_npz(f'./generated_dataset/{cfg.data.name}/{embedding_model_name}_{cfg.seed}_train_dataset.npz', train_dataset)
        # print(f'Saved train dataset to ./generated_dataset/{cfg.data.name}/{embedding_model_name}_{cfg.seed}_train_dataset.npz')
        # print(f'save data: {time.time() - start_time:.2f} seconds')
        # torch.save(train_labels, f'./generated_dataset/{cfg.data.name}/{embedding_model_name}_{cfg.seed}_train_labels.npz')
        # print(f'Saved train labels to ./generated_dataset/{cfg.data.name}/{embedding_model_name}_{cfg.seed}_train_labels.npz')
        # print(f'save label: {time.time() - start_time:.2f} seconds')
        
        # ssp.save_npz(f'./generated_dataset/{cfg.data.name}/{embedding_model_name}_{cfg.seed}_val_dataset.npz', val_dataset)
        # print(f'save data: {time.time() - start_time:.2f} seconds')
        # print(f'Saved validation dataset to ./generated_dataset/{cfg.data.name}/{embedding_model_name}_{cfg.seed}_val_dataset.npz')
        # torch.save(val_labels, f'./generated_dataset/{cfg.data.name}/{embedding_model_name}_{cfg.seed}_val_labels.npz')
        # print(f'save label: {time.time() - start_time:.2f} seconds')
        # print(f'Saved validation labels to ./generated_dataset/{cfg.data.name}/{embedding_model_name}_{cfg.seed}_val_labels.npz')
        # del val_dataset
            
        # ssp.save_npz(f'./generated_dataset/{cfg.data.name}/{embedding_model_name}_{cfg.seed}_test_dataset.npz', test_dataset)
        # print(f'save data: {time.time() - start_time:.2f} seconds')
        # print(f'Saved test dataset to ./generated_dataset/{cfg.data.name}/{embedding_model_name}_{cfg.seed}_test_dataset.npz')
        # torch.save(test_labels, f'./generated_dataset/{cfg.data.name}/{embedding_model_name}_{cfg.seed}_test_labels.npz')
        # print(f'save label: {time.time() - start_time:.2f} seconds')
        # print(f'Saved test labels to ./generated_dataset/{cfg.data.name}/{embedding_model_name}_{cfg.seed}_test_labels.npz')
        # del test_dataset

    
    
def list2csr(lst: List):
    # Identify non-zero values and their positions
    data = []
    indices = []
    for idx, value in enumerate(lst):
        if value != 0:
            data.append(value)
            indices.append(idx)

    # Create CSR matrix
    sparse_matrix = scipy.sparse.csr_matrix((data, indices, [0, len(indices)]), shape=(1, len(lst)))

    return sparse_matrix

def create_tfidf(cfg, seed):
    seed_everything(seed)
    splits, text, _ = load_data_lp[cfg.data.name](cfg.data)
    if cfg.embedder.type == 'tfidf':
        train_dataset, train_labels = process_texts(
            splits['train'].pos_edge_label_index, 
            splits['train'].neg_edge_label_index, 
            text
        )
        val_dataset, val_labels = process_texts(
            splits['valid'].pos_edge_label_index, 
            splits['valid'].neg_edge_label_index, 
            text
        )
        test_dataset, test_labels = process_texts(
            splits['test'].pos_edge_label_index, 
            splits['test'].neg_edge_label_index, 
            text
        )
        vectorizer = TfidfVectorizer()
        
        os.makedirs(f'./generated_dataset/{cfg.data.name}/', exist_ok=True)
        start_time = time.time()
        train_data = vectorizer.fit_transform(train_dataset)
        print(f'fit_transform: {time.time() - start_time:.2f} seconds')

        # del train_dataset
        start_time = time.time()
        val_data = vectorizer.transform(val_dataset)
        print(f'fit_transform: {time.time() - start_time:.2f} seconds')
        
        start_time = time.time()
        test_data = vectorizer.transform(test_dataset)
        print(f'fit_transform: {time.time() - start_time:.2f} seconds')
        
    return train_data, train_labels, val_data, val_labels, test_data, test_labels


if __name__ == "__main__":
    
    file_path = f'{get_git_repo_root_path()}/'
    args = parse_args()
    cfg = set_cfg(file_path, args.cfg_file)
    cfg.merge_from_list(args.opts)
    # custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)
    cfg = config_device(cfg)
    cfg.data.name = args.data
    cfg.seed = args.seed
    
    embedding_model_name = "tfidf"
    create_tfidf(embed_type, cfg, args)
    train_dataset, train_labels, val_dataset, val_labels, test_dataset, test_labels = save_dataset(embedding_model_name, cfg, args)
