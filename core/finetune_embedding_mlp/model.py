import os
import sys

import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from utils import init_random_state

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


import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool


class TNPClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, loss_func, dropout=0.0, seed=0, cla_bias=True):
        super().__init__(model.config)
        self.encoder, self.loss_func = model, loss_func
        self.dropout = nn.Dropout(dropout)
        hidden_dim = model.config.hidden_size
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        init_random_state(seed)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # Extract outputs from the model
        outputs = self.encoder(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        emb = self.dropout(outputs['hidden_states'][-1])  # outputs[0]=last hidden state
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        logits = self.classifier(cls_token_emb)
        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)
        return TokenClassifierOutput(loss=loss, logits=logits)


class BertClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, loss_func, dropout=0.0, seed=0, cla_bias=True):
        super().__init__(model.config)
        self.bert_encoder, self.loss_func = model, loss_func
        self.dropout = nn.Dropout(dropout)
        hidden_dim = model.config.hidden_size
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        init_random_state(seed)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the model
        outputs = self.bert_encoder(input_ids, attention_mask, output_hidden_states=True)
        emb = self.dropout(outputs['hidden_states'][-1])  # outputs[0]=last hidden state
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        logits = self.classifier(cls_token_emb)
        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)
        return TokenClassifierOutput(loss=loss, logits=logits)


class BertEmbInfModel(PreTrainedModel):
    def __init__(self, model):
        super().__init__(model.config)
        self.bert_encoder = model

    @torch.no_grad()
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # Extract outputs from the model
        outputs = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask,
                                    output_hidden_states=True)
        emb = outputs['hidden_states'][-1]  # Last layer
        # Use CLS Emb as sentence emb.
        node_cls_emb = emb.permute(1, 0, 2)[0]
        return TokenClassifierOutput(logits=node_cls_emb)


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


def infonce(anchor, sample, tau=0.2):
    sim = _similarity(anchor, sample) / tau
    num_nodes = anchor.shape[0]
    device = anchor.device
    pos_mask = torch.eye(num_nodes, dtype=torch.float32).to(device)
    neg_mask = 1. - pos_mask
    assert sim.size() == pos_mask.size()  # sanity check
    exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
    return -loss.mean()