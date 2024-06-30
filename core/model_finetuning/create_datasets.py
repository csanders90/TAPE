import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec, KeyedVectors
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from graphgps.utility.utils import (
    set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir,
    custom_set_run_dir, set_printing, run_loop_settings, create_optimizer,
    config_device, init_model_from_pretrained, create_logger, use_pretrained_llm_embeddings
)

from torch_geometric.graphgym.config import dump_cfg, makedirs_rm_exist
from data_utils.load import load_data_nc, load_data_lp

def get_word2vec_embeddings(model, text):
    words = text.split()
    if word_vectors := [model.wv[word] for word in words if word in model.wv]:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def get_embeddings(text, embedding_model, embedding_model_type):
    if embedding_model_type == "tfidf" or embedding_model_type != "word2vec":
        return embedding_model.encode(text)
    else:
        return get_word2vec_embeddings(embedding_model, text)

def get_tf_idf_by_texts(texts, known_mask, test_mask, max_features=1433, use_tokenizer=False):
    if known_mask is None and test_mask is None:
        tf_idf_vec = TfidfVectorizer(stop_words='english', max_features=max_features)
        X = tf_idf_vec.fit_transform(texts)
        torch_feat = torch.FloatTensor(X.todense())
        norm_torch_feat = F.normalize(torch_feat, dim=-1)
        return torch_feat, norm_torch_feat
    
    if use_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="/tmp")
        tf_idf_vec = TfidfVectorizer(analyzer="word", max_features=500, tokenizer=lambda x: tokenizer.tokenize(x, max_length=512, truncation=True))
    else:
        tf_idf_vec = TfidfVectorizer(stop_words='english', max_features=max_features)

    text_known = [texts[i] for i in range(len(texts)) if known_mask[i]]
    text_test = [texts[i] for i in range(len(texts)) if test_mask[i]]

    x_known = tf_idf_vec.fit_transform(text_known)
    x_test = tf_idf_vec.transform(text_test)

    x_known = torch.FloatTensor(x_known.todense())
    x_test = torch.FloatTensor(x_test.todense())

    dim = x_known.shape[1]
    torch_feat = torch.zeros(len(texts), dim)
    torch_feat[known_mask] = x_known 
    torch_feat[test_mask] = x_test

    norm_torch_feat = F.normalize(torch_feat, dim=-1)
    return torch_feat, norm_torch_feat


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
        combined_text = f"{text1} {text2}"
        dataset.append(combined_text)
        labels.append(1)

    # Process negative edges
    for i in tqdm(range(neg_edge_index.shape[1])):
        node1 = neg_edge_index[0, i].item()
        node2 = neg_edge_index[1, i].item()
        text1 = text[node1]
        text2 = text[node2]
        combined_text = f"{text1} {text2}"
        dataset.append(combined_text)
        labels.append(0)

    return dataset, labels

def norm_feat(train_text: np.array):
    X = tf_idf_vec.fit_transform(train_text)
    torch_feat = torch.FloatTensor(X.todense())
    return F.normalize(torch_feat, dim=-1)

if __name__ == '__main__':
    embedding_model_name = "tfidf"
    

    FILE_PATH = f'{get_git_repo_root_path()}/'

    args = parse_args()
    # Load args file

    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)
    custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)
    dump_cfg(cfg)

    # # Set Pytorch environment
    torch.set_num_threads(cfg.run.num_threads)

    loggers = create_logger(args.repeat)
    cfg = config_device(cfg)
    splits, text, data = load_data_lp[cfg.data.name](cfg.data)

    max_features = data.x.shape[1]
    
    dataset = []
    pos_train_edge_index = splits['train'].pos_edge_label_index
    neg_train_edge_index = splits['train'].neg_edge_label_index

    pos_val_edge_index = splits['valid'].pos_edge_label_index
    neg_val_edge_index = splits['valid'].neg_edge_label_index

    pos_test_edge_index = splits['test'].pos_edge_label_index
    neg_test_edge_index = splits['test'].neg_edge_label_index

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if embedding_model_name == "tfidf":
        train_text, train_labels = process_texts(
            splits['train'].pos_edge_label_index, 
            splits['train'].neg_edge_label_index, 
            text
        )
        val_text, val_labels = process_texts(
            splits['valid'].pos_edge_label_index, 
            splits['valid'].neg_edge_label_index, 
            text
        )
        test_text, test_labels = process_texts(
            splits['test'].pos_edge_label_index, 
            splits['test'].neg_edge_label_index, 
            text
        )
        tf_idf_vec = TfidfVectorizer(stop_words='english', max_features=max_features)
        train_dataset = norm_feat(train_text)
        val_dataset = norm_feat(val_text)
        test_dataset = norm_feat(test_text)

        print("Normalized TF-IDF Features:\n", test_dataset)
        print("Normalized TF-IDF Features:\n", val_dataset)
        print("Normalized TF-IDF Features:\n", train_dataset)
        
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

    # Convert to tensors
    train_dataset = torch.tensor(train_dataset, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_dataset = torch.tensor(val_dataset, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    test_dataset = torch.tensor(test_dataset, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Save datasets
    torch.save(train_dataset, f'./generated_dataset/{embedding_model_name}_train_dataset.pt')
    torch.save(train_labels, f'./generated_dataset/{embedding_model_name}_train_labels.pt')
    torch.save(val_dataset, f'./generated_dataset/{embedding_model_name}_val_dataset.pt')
    torch.save(val_labels, f'./generated_dataset/{embedding_model_name}_val_labels.pt')
    torch.save(test_dataset, f'./generated_dataset/{embedding_model_name}_test_dataset.pt')
    torch.save(test_labels, f'./generated_dataset/{embedding_model_name}_test_labels.pt')


    # Example usage
    known_mask = None 
    test_mask = None
    torch_feat, norm_torch_feat = get_tf_idf_by_texts(train_text, known_mask, test_mask, max_features=10, use_tokenizer=False)

    print("TF-IDF Features:\n", torch_feat)
    print("Normalized TF-IDF Features:\n", norm_torch_feat)
