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
from typing import List, Tuple
import scipy
import argparse
import time 
import nltk
from nltk.tokenize import word_tokenize
import re
from graphgps.utility.utils import random_sampling

nltk.download('punkt')


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
    

def process_texts(pos_edge_index: torch.tensor, neg_edge_index: torch.tensor, text) -> Tuple[List[np.ndarray], np.ndarray]:
    dataset = []
    labels = []
    # Process positive edges
    for i in tqdm(range(pos_edge_index.shape[1])):
        node1 = pos_edge_index[0, i].item()
        node2 = pos_edge_index[1, i].item()
        text1 = text[node1]
        text2 = text[node2]
        combined_text = text1 + "," + text2
        dataset.append(combined_text)
        labels.append(1)

    # Process negative edges
    for i in tqdm(range(neg_edge_index.shape[1])):
        node1 = neg_edge_index[0, i].item()
        node2 = neg_edge_index[1, i].item()
        text1 = text[node1]
        text2 = text[node2]
        combined_text = text1 + "," + text2
        dataset.append(combined_text)
        labels.append(0)
    
    return dataset, np.array(labels)


def process_nodefeat(pos_edge_index, neg_edge_index, text):
    dataset = []
    labels = []
    
    # Process positive edges
    for i in tqdm(range(pos_edge_index.shape[1])):
        node1 = pos_edge_index[0, i].item()
        node2 = pos_edge_index[1, i].item()
        text1 = text[node1].numpy()
        text2 = text[node2].numpy()
        combined_text = np.concatenate((text1, text2), axis=0)
        dataset.append(combined_text)
        labels.append(1)

    # Process negative edges
    for i in tqdm(range(neg_edge_index.shape[1])):
        node1 = neg_edge_index[0, i].item()
        node2 = neg_edge_index[1, i].item()
        text1 = text[node1].numpy()
        text2 = text[node2].numpy()
        combined_text = np.concatenate((text1, text2), axis=0)
        dataset.append(combined_text)
        labels.append(0)
    
    return np.array(dataset), np.array(labels)


def load_or_generate_datasets(splits, node_features, run_id, cfg):
    

    data_folder = 'generated_dataset/'
    train_data_path = data_folder + f'train_data_{run_id}_{cfg.data.name}_{cfg.embedder.type}.pth'
    train_labels_path = data_folder + f'train_labels_{run_id}_{cfg.data.name}_{cfg.embedder.type}.pth'
    val_data_path = data_folder + f'val_data_{run_id}_{cfg.data.name}_{cfg.embedder.type}.pth'
    val_labels_path = data_folder + f'val_labels_{run_id}_{cfg.data.name}_{cfg.embedder.type}.pth'
    test_data_path = data_folder + f'test_data_{run_id}_{cfg.data.name}_{cfg.embedder.type}.pth'
    test_labels_path = data_folder + f'test_labels_{run_id}_{cfg.data.name}_{cfg.embedder.type}.pth'


    if (os.path.exists(train_data_path) and os.path.exists(train_labels_path) and
        os.path.exists(val_data_path) and os.path.exists(val_labels_path) and
        os.path.exists(test_data_path) and os.path.exists(test_labels_path)):
        
        start_time = time.time()
        train_dataset = torch.load(train_data_path)
        train_labels = torch.load(train_labels_path)
        val_dataset = torch.load(val_data_path)
        val_labels = torch.load(val_labels_path)
        test_dataset = torch.load(test_data_path)
        test_labels = torch.load(test_labels_path)
        print(f"Time taken to load the dataset: {time.time() - start_time}")
        
    else:
        train_dataset, train_labels = process_nodefeat(
            splits['train'].pos_edge_label_index, 
            splits['train'].neg_edge_label_index, 
            node_features
        )
        val_dataset, val_labels = process_nodefeat(
            splits['valid'].pos_edge_label_index, 
            splits['valid'].neg_edge_label_index, 
            node_features
        )
        test_dataset, test_labels = process_nodefeat(
            splits['test'].pos_edge_label_index, 
            splits['test'].neg_edge_label_index, 
            node_features
        )
        
        os.makedirs(data_folder, exist_ok=True)
        torch.save(train_dataset, train_data_path)
        torch.save(train_labels, train_labels_path)
        torch.save(val_dataset, val_data_path)
        torch.save(val_labels, val_labels_path)
        torch.save(test_dataset, test_data_path)
        torch.save(test_labels, test_labels_path)

    return train_dataset, train_labels, val_dataset, val_labels, test_dataset, test_labels




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
                    
            # Function to get the average embedding for a whole text (e.g., title and abstract combined)
            def get_average_embedding(text, model):
                tokens = preprocess(text)
                embeddings = [model.wv[token] for token in tokens if token in model.wv]
                if embeddings:
                    return np.mean(embeddings, axis=0)
                else:
                    # Return a zero vector if none of the tokens are in the vocabulary
                    return np.zeros(model.vector_size)

            # Example usage

            def preprocess(text):
                # Remove non-alphanumeric characters
                text = re.sub(r'\W+', ' ', text)
                # Tokenize and convert to lowercase
                tokens = word_tokenize(text.lower())
                return tokens
            
            tokenized_texts = [preprocess(text) for text in text]

            # Train a Word2Vec model
                
            model = Word2Vec(sentences=tokenized_texts, vector_size=128, window=5, min_count=1, workers=10)

            w2v_nodefeat = np.array([get_average_embedding(text, model) for text in text])

            train_dataset, train_labels = process_edges(
                splits['train'].pos_edge_label_index, 
                splits['train'].neg_edge_label_index, 
                text,
            )
            val_dataset, val_labels = process_edges(
                splits['valid'].pos_edge_label_index, 
                splits['valid'].neg_edge_label_index, 
                text, 
            )
            test_dataset, test_labels = process_edges(
                splits['test'].pos_edge_label_index, 
                splits['test'].neg_edge_label_index, 
                text, 
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

    sparse_matrix = scipy.sparse.csr_matrix((data, indices, [0, len(indices)]), shape=(1, len(lst)))

    return sparse_matrix



 
def create_tfidf(cfg, seed):
    seed_everything(seed)
    cfg.data.method = cfg.embedder.type
    splits, text, _ = load_data_lp[cfg.data.name](cfg.data)
    splits = random_sampling(splits, cfg.data.scale)
    
    if cfg.data.name in ['pwc_small', 'pwc_medium', 'pwc_large']:
        text = text['feat'].tolist()

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
        
    if cfg.embedder.type == 'tfidf':

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
    
    elif cfg.embedder.type == 'w2v':
        def get_average_embedding(text, model):
            tokens = preprocess(text)
            embeddings = [model.wv[token] for token in tokens if token in model.wv]
            if embeddings:
                return np.mean(embeddings, axis=0)
            else:
                return np.zeros(model.vector_size)

        def preprocess(text):
            text = re.sub(r'\W+', ' ', text)
            tokens = word_tokenize(text.lower())
            return tokens
        
        tokenized_texts = [preprocess(text) for text in text]
        model = Word2Vec(sentences=tokenized_texts, vector_size=128, window=5, min_count=1, workers=10)
        train_data = np.array([get_average_embedding(text, model) for text in train_dataset])
        val_data = np.array([get_average_embedding(text, model) for text in val_dataset])
        test_data = np.array([get_average_embedding(text, model) for text in test_dataset])

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


if __name__ == "__main__":
    
    file_path = f'{get_git_repo_root_path()}/'
    args = parse_args()
    cfg = set_cfg(file_path, args.cfg_file)
    cfg.merge_from_list(args.opts)
    
    cfg = config_device(cfg)
    cfg.data.name = args.data
    cfg.seed = args.seed
    
    embedding_model_name = "tfidf"
    create_tfidf(embedding_model_name, cfg, args)
    train_dataset, train_labels, val_dataset, val_labels, test_dataset, test_labels = save_dataset(embedding_model_name, cfg, args)
