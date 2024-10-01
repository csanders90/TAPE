import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from data_utils.load import load_data_lp
from typing import Dict, Any, Tuple
from datasets import Dataset
from transformers import AutoTokenizer
from make_trainer import ModelBuilder, ModelTrainer
import yaml
import torch

if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")

def tokenize_data(batch: Dict[str, Any], model_name=str) -> Tuple[Dataset, Dataset]:
    """
    Tokenize a batch of data
    
    Args:
        batch (dict): batch of texts
        

    Returns:
        dict: tokenized batch of data
    """
    # Load configs

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(batch["text"], padding=True, truncation=True)

def preprocess_data(pos_edges, neg_edges):
    dataset = []

    # Process positive edges
    for i in range(pos_edges.shape[1]):
        node1 = pos_edges[0, i].item()
        node2 = pos_edges[1, i].item()
        text1 = text[node1]
        text2 = text[node2]
        text_example = text1 + "\n\n" + text2
        dataset.append([text_example, 1])

    # Process negative edges
    for i in range(neg_edges.shape[1]):
        node1 = neg_edges[0, i].item()
        node2 = neg_edges[1, i].item()
        text1 = text[node1]
        text2 = text[node2]
        text_example = text1 + "\n\n" + text2
        dataset.append([text_example, 0])

    tokenized_data = dataset.map(tokenize_data, batched=True)
    return tokenized_data

def load_config(config_path: str):
    """
    Loads the configuration for the respective training task

    Arguments:
        config_path: path to config file
    """

    with open(config_path, encoding="utf-8") as file:
        return yaml.safe_load(file)

def parse_arguments():
    """
    Parse arguments when running fine-tuning script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./src/use_cases/prompt_injection_classification/config.yaml",
        help="Path to the configuration file for the training task."
    )

    args = parser.parse_args()
    return args
    

args = parse_arguments()
fine_tuning_config = load_config("fine_tuning_config.yaml")
run_name = "fine_tuning_test"

splits, text, data = load_data_lp["cora"]("cora")

pos_train_edge_index = splits['train'].pos_edge_label_index
neg_train_edge_index = splits['train'].neg_edge_label_index

pos_val_edge_index = splits['val'].pos_edge_label_index
neg_val_edge_index = splits['val'].neg_edge_label_index

pos_test_edge_index = splits['test'].pos_edge_label_index
neg_test_edge_index = splits['test'].neg_edge_label_index

train_data = tokenize_data(pos_train_edge_index, neg_train_edge_index)
val_data = tokenize_data(pos_val_edge_index, neg_val_edge_index)
test_data = tokenize_data(pos_test_edge_index, neg_test_edge_index)

num_classes = 2
trainer = ModelTrainer(fine_tuning_config, run_name)
model_builder = ModelBuilder(fine_tuning_config, num_classes)
model = model_builder.build_model()

# training
trained_model, train_time = trainer.train(train_data, val_data, model)

# evaluation on test data
test_results = trained_model.evaluate(test_data)
print(test_results)


