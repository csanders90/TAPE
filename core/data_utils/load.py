import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import csv
from data_utils.dataset import CustomDGLDataset
from data_utils.load_cora_lp import get_cora_casestudy 
from data_utils.load_arxiv_2023_lp import get_raw_text_arxiv_2023
from data_utils.load_pubmed_lp import get_pubmed_casestudy
from data_utils.load_ogbn_arxiv import get_raw_text_ogbn_arxiv_lp
from data_utils.load_products import get_raw_text_products_lp

data_loader = {
    'cora': get_cora_casestudy,
    'pubmed': get_pubmed_casestudy,
    'arxiv_2023': get_raw_text_arxiv_2023,
    'ogbn-arxiv': get_raw_text_ogbn_arxiv_lp,
    'ogbn-products': get_raw_text_products_lp,
}

def load_gpt_preds(dataset, topk):
    preds = []
    fn = f'gpt_preds/{dataset}.csv'
    print(f"Loading topk preds from {fn}")
    with open(fn, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            inner_list = []
            for value in row:
                inner_list.append(int(value))
            preds.append(inner_list)

    pl = torch.zeros(len(preds), topk, dtype=torch.long)
    for i, pred in enumerate(preds):
        pl[i][:len(pred)] = torch.tensor(pred[:topk], dtype=torch.long)+1
    return pl


def load_data(dataset, use_dgl=False, use_text=False, use_gpt=False, seed=0):
    if dataset == 'cora':
        from data_utils.load_cora import get_raw_text_cora as get_raw_text
        num_classes = 7
    elif dataset == 'pubmed':
        from data_utils.load_pubmed import get_raw_text_pubmed as get_raw_text
        num_classes = 3
    elif dataset == 'ogbn-arxiv':
        from core.data_utils.load_ogbn_arxiv import get_raw_text_arxiv as get_raw_text
        num_classes = 40
    elif dataset == 'ogbn-products':
        from data_utils.load_products import get_raw_text_products as get_raw_text
        num_classes = 47
    elif dataset == 'arxiv_2023':
        from data_utils.load_arxiv_2023 import get_raw_text_arxiv_2023 as get_raw_text
        num_classes = 40
    else:
        exit(f'Error: Dataset {dataset} not supported')

    # for training GNN
    if use_text:
        data, text = get_raw_text(use_text, seed=seed)
        if use_dgl:
            data = CustomDGLDataset(dataset, data)
        return data, num_classes, text

    # for finetuning LM
    if use_gpt:
        data, text = get_raw_text(use_text=False, seed=seed)
        folder_path = 'gpt_responses/{}'.format(dataset)
        print(f"using gpt: {folder_path}")
        n = data.y.shape[0]
        text = []
        for i in range(n):
            filename = str(i) + '.json'
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                content = json_data['choices'][0]['message']['content']
                text.append(content)
    # if use_gpt:
    #     data, text = get_raw_text(use_text=False, seed=seed)
    #     folder_path = 'llama_responses/{}'.format(dataset)
    #     print(f"using gpt: {folder_path}")
    #     n = data.y.shape[0]
    #     text = []
    #     for i in range(n):
    #         filename = str(i) + '.json'
    #         file_path = os.path.join(folder_path, filename)
    #         with open(file_path, 'r') as file:
    #             json_data = json.load(file)
    #             content = json_data['generation']['content']
    #             text.append(content)
    else:
        data, text = get_raw_text(use_text=True, seed=seed)

    return data, num_classes, text
