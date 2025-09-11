# -*- coding: utf-8 -*-
import torch
import copy
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

from ..Mamba import MambaBlock as mamba
from ..Fusion import ProjectFusion

torch.backends.cudnn.enabled = False

class MyModel(SequentialRecommender):
    def __init__(self, config, dataset):
        super(MyModel, self).__init__(config, dataset)

        # load parameters info
        self.max_len = config["MAX_ITEM_LIST_LENGTH"]
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.dropout_prob = config["hidden_dropout_prob"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.d_state = config["d_state"]
        self.c = config["c"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

        self.mamba1=mamba(d_model=self.hidden_size, d_state=self.d_state, d_conv=2, expand=1) 
        self.mamba2=mamba(d_model=self.max_len, d_state=self.d_state, d_conv=2, expand=1) 
        self.fusion = ProjectFusion()

        self.linear_out = nn.Linear(self.hidden_size, self.embedding_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.high_gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size,
                      batch_first=True, bidirectional=False)


    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        res = item_seq_emb
        
        # 拆分频域
        batch, seq_len, hidden = item_seq_emb.shape
        x = torch.fft.rfft(item_seq_emb, dim=1, norm='ortho')
        low_pass = x[:]
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = item_seq_emb - low_pass

        # ➤ GRU 分支处理 low_pass
        high_output, _ = self.high_gru(low_pass)  # 输出 shape: [B, L, H]
        

        # ➤ Mamba 分支处理 high_pass
        output1 = self.mamba1(high_pass)

        output2 = torch.permute(high_pass,(0,2,1)) # B, E, L
        output2 = self.mamba2(output2) # B, E, L
        output2 = torch.permute(output2,(0,2,1)) # B, L, E

        # ➤ 融合输出
        output = self.fusion(output1 + output2, high_output)

        output = self.LayerNorm(self.dropout(self.linear_out(output) + res))
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        return seq_output


    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores



