import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils.load import load_data_lp
from utils import set_cfg, parse_args, get_git_repo_root_path
from sentence_transformers import SentenceTransformer
from typing import List
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
import re
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from openai import OpenAI
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

FILE_PATH = f'{get_git_repo_root_path()}/'

args = parse_args()

cfg = set_cfg(FILE_PATH, args.cfg_file)
cfg.merge_from_list(args.opts)

splits, text = load_data_lp[cfg.data.name](cfg.data)


def create_sentence_embeddings(model_name: str, data: List[str], device: str, with_preprocessing: bool=False):
    embedding_model = SentenceTransformer(model_name)
    print("Start sentence embedding generation")
    embeddings = torch.tensor(embedding_model.encode(data))
    print("Embedding sentence generation completed")
    return embeddings

def create_llm_embeddings_from_pretrained(model_name: str, data: List[str], device: str, token: str, with_preprocessing: bool=False, batch_size: int=8):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    device_ids = [i for i in range(torch.cuda.device_count())]
    model = AutoModel.from_pretrained(model_name, token=token)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    encoded_input = tokenizer(data, padding=True, truncation=True, return_tensors='pt').to(device)
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']
    
    # Create a TensorDataset and DataLoader for batching
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    all_embeddings = []
    
    model.eval() 
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_input_ids, batch_attention_mask = [b.to(device) for b in batch]
            model_output = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            sentence_embeddings = mean_pooling(model_output, batch_attention_mask)
            normalized_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            all_embeddings.append(normalized_embeddings)
        
    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings

def create_openai_embeddings(model_name: str, open_ai_api_key: str, data = List[str]) -> torch.Tensor:
    open_ai_client = OpenAI(api_key=open_ai_api_key)
    embeddings = [] 
    print("Start OpenAI embedding generation")
    for text in tqdm(data):
        response = open_ai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        embeddings.append(response.data[0].embedding)
    embeddings = torch.tensor(embeddings)
    print("Embedding generation completed")
    return embeddings

def create_tfidf_embeddings(data: List[str], with_preprocessing: bool=False):
    vectorizer = TfidfVectorizer()
    print("Start tfidf embedding generation")
    
    if with_preprocessing==True:
        data = [text_preprocessing(text) for text in data]
    embeddings = torch.tensor(vectorizer.fit_transform(data).toarray(), dtype=torch.float32)
    print("Embedding generation completed")
    return embeddings

def text_preprocessing(text: str):

    #lower
    text = text.lower()
    
    # remove numbers
    text = re.sub(r'[0-9]+', '', text)
    
    #stemming
    stemmer = SnowballStemmer("english")
    stemmed_words = [stemmer.stem(word) for word in text.split()]
    text = ' '.join(stemmed_words)
    return text
    
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
