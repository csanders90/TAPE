import os
import sys

import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from utils import init_random_state
from typing import Tuple, List, Dict, Any, Union


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graphgps.score.custom_score import LinkPredictor


class BertClassifier(PreTrainedModel):
    def __init__(self, model, cfg, dropout=0.0, seed=0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(
                model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = LinkPredictor(hidden_dim, cfg.model.hidden_channels, 1, cfg.model.num_layers,
                                        cfg.model.dropout, 'dot')
        init_random_state(seed)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                preds=None):
        input_1 = input_ids[:, 0, :]
        input_2 = input_ids[:, 1, :]
        attention_mask_1 = attention_mask[:, 0, :]
        attention_mask_2 = attention_mask[:, 1, :]

        outputs_1 = self.bert_encoder(input_ids=input_1,
                                      attention_mask=attention_mask_1,
                                      return_dict=return_dict,
                                      output_hidden_states=True)
        outputs_2 = self.bert_encoder(input_ids=input_2,
                                      attention_mask=attention_mask_2,
                                      return_dict=return_dict,
                                      output_hidden_states=True)
        # outputs[0]=last hidden state
        emb_1 = self.dropout(outputs_1['hidden_states'][-1])
        emb_2 = self.dropout(outputs_2['hidden_states'][-1])
        # Use CLS Emb as sentence emb.
        cls_token_emb_1 = emb_1.permute(1, 0, 2)[0]
        cls_token_emb_2 = emb_2.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb_1 = self.feat_shrink_layer(cls_token_emb_1)
            cls_token_emb_2 = self.feat_shrink_layer(cls_token_emb_2)

        logits = self.classifier(cls_token_emb_1, cls_token_emb_2)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()

        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        pos_out = logits[pos_mask]
        neg_out = logits[neg_mask]

        pos_loss = -torch.log(pos_out + 1e-15).mean() if pos_out.numel() > 0 else torch.tensor(0.0)

        neg_loss = -torch.log(1 - neg_out + 1e-15).mean() if neg_out.numel() > 0 else torch.tensor(0.0)

        loss = pos_loss + neg_loss
        return TokenClassifierOutput(loss=loss, logits=logits)


class BertClaInfModel(PreTrainedModel):
    def __init__(self, model, emb, pred, feat_shrink=''):
        super().__init__(model.config)
        self.bert_classifier = model
        self.emb, self.pred = emb, pred
        self.feat_shrink = feat_shrink

    @torch.no_grad()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                node_id=None):

        # Extract outputs from the model
        input_1 = input_ids[:, 0, :]
        input_2 = input_ids[:, 1, :]
        attention_mask_1 = attention_mask[:, 0, :]
        attention_mask_2 = attention_mask[:, 1, :]

        outputs_1 = self.bert_classifier.bert_encoder(input_ids=input_1,
                                      attention_mask=attention_mask_1,
                                      return_dict=return_dict,
                                      output_hidden_states=True)
        outputs_2 = self.bert_classifier.bert_encoder(input_ids=input_2,
                                      attention_mask=attention_mask_2,
                                      return_dict=return_dict,
                                      output_hidden_states=True)
        # outputs[0]=last hidden state
        emb_1 = outputs_1['hidden_states'][-1]
        emb_2 = outputs_2['hidden_states'][-1]
        # Use CLS Emb as sentence emb.
        cls_token_emb_1 = emb_1.permute(1, 0, 2)[0]
        cls_token_emb_2 = emb_2.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb_1 = self.feat_shrink_layer(cls_token_emb_1)
            cls_token_emb_2 = self.feat_shrink_layer(cls_token_emb_2)

        logits = self.bert_classifier.classifier(cls_token_emb_1, cls_token_emb_2).squeeze(dim=1)
        if torch.cuda.is_available():
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        self.emb = torch.stack((cls_token_emb_1, cls_token_emb_2), dim=1).cpu().numpy().astype(np.float16)
        self.pred = logits.cpu().numpy().astype(np.float16)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        pos_out = logits[pos_mask]
        neg_out = logits[neg_mask]

        pos_loss = -torch.log(pos_out + 1e-15).mean() if pos_out.numel() > 0 else torch.tensor(0.0)

        neg_loss = -torch.log(1 - neg_out + 1e-15).mean() if neg_out.numel() > 0 else torch.tensor(0.0)

        loss = pos_loss + neg_loss
        return TokenClassifierOutput(loss=loss, logits=logits)



class Co_LMGCN(PreTrainedModel):
    # add initialized gnn as a input
    def __init__(self, model, cfg, GNN, dropout=0.0, seed=0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder = model #LLM 
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(
                model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        
        # add params for gcns
        self.gnn = GNN # GNN 
        hidden_dim = GNN.n_hidden
        
        self.classifier = LinkPredictor(hidden_dim, cfg.model.hidden_channels, 1, cfg.model.num_layers,
                                        cfg.model.dropout, 'dot')
        init_random_state(seed)


    
    def forward(self,
                g: torch.tensor,
                input_ids: torch.tensor=None,
                attention_mask: torch.tensor=None,
                node_id: torch.tensor=None,
                labels=None,
                return_dict=None,
                preds=None):
        input_1 = input_ids[:, 0, :]
        attention_mask_1 = attention_mask[:, 0, :]

        outputs_1 = self.bert_encoder(input_ids=input_1,
                                      attention_mask=attention_mask_1,
                                      return_dict=return_dict,
                                      output_hidden_states=True)
        # outputs[0]=last hidden state
        emb_1 = self.dropout(outputs_1['hidden_states'][-1])

        # Use CLS Emb as sentence emb.
        cls_token_emb_1 = emb_1.permute(1, 0, 2)[0]
        if self.feat_shrink:
            text_emb = self.feat_shrink_layer(cls_token_emb_1)
        # cls_token_emb_1 should be the input node features for GNN
        # g should be the adjacency matrix
        x = self.gnn(g, text_emb)
        # node_id is a tensor of shape (2, 1), where first element is the index
        # of the source node,and second element is the idx of the target node
        logits = self.classifier(x[node_id[0]], x[node_id[1]])

        if labels.shape[-1] == 1:
            labels = labels.squeeze()

        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        pos_out = logits[pos_mask]
        neg_out = logits[neg_mask]

        pos_loss = -torch.log(pos_out + 1e-15).mean() if pos_out.numel() > 0 else torch.tensor(0.0)

        neg_loss = -torch.log(1 - neg_out + 1e-15).mean() if neg_out.numel() > 0 else torch.tensor(0.0)

        loss = pos_loss + neg_loss
        return TokenClassifierOutput(loss=loss, logits=logits)
