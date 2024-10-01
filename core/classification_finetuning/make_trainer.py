from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Any, Dict, Tuple
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import time
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelBuilder:
    """
    Class respobsible for building and configuring the model for training

    Attributes:
        config (dict): Configuration loaded from yaml file
    """

    def __init__(self, config: Dict[str, Any], num_classes: int):
        """
        Initialize model builder with configuration

        Args:
            config (dict): Configuration loaded from yaml file
        """
        self.config = config
        self.num_classes = num_classes

    def build_model(self) -> AutoModelForSequenceClassification:
        """
        Build the model and apply quantization configurations

        Returns:
            AutoModelForSequenceClassification: the configured model
        """

        self.config["bnb_4bit_compute_dtype"] = torch.bfloat16

        # load quantization config for the model
        bnb_config = BitsAndBytesConfig(**self.config["bnb_config"])

        # load sequence classification model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config["model_name"],
            config=AutoConfig.from_pretrained(self.config["model_name"], num_labels=self.num_classes),
            quantization_config=bnb_config,
            device_map="auto"
        )

        # prepare model for quantization
        model = prepare_model_for_kbit_training(model)

        # Apply LoRA 
        lora_config = LoraConfig(**self.config["lora_config"])
        model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()
        return model
    

class ModelTrainer:
    """
    Handling data loading, tokenization, training and evaluation

    Attributes:
        config (dict): config loaded from yaml file
        tokenizer (AutoTokenizer): tokenizer for the model
    """
    def __init__(self, config: Dict[str, Any], run_name: str):
        self.config = config
        self.run_name = run_name
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        # Calculate accuracy
        accuracy = accuracy_score(labels, preds)

        # Calculate precision, recall, and F1-score
        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        f1 = f1_score(labels, preds, average='macro')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
            
    def train(self, tokenized_train_data: Dataset, tokenized_val_data: Dataset, model: Any) -> Tuple[Trainer, float]:
        """
        Training of the model

        Args:
            tokenized_train_data (Dataset): train data
            tokenized_val_data (Dataset): val data
            model (PreTrainedModel): model to train

        Returns:
            tuple: trained model and training time
        """

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.config["training_args"]["output_dir"] = self.config["training_args"]["output_dir"] + self.run_name

        training_args = TrainingArguments(
            **self.config["training_args"],
            label_names=["labels"]
            )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_data,
            eval_dataset=tokenized_val_data,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        start_time = time.time()
        trainer.train()
        end_time = time.time()
        training_time = end_time - start_time

        return trainer, training_time
    