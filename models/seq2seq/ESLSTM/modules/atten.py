# -*- coding: utf-8 -*-
"""
@author: Wendong;
@ARTICLE{HSN_LSTM, 
        author={Zheng, Wendong and Zhao, Putian and Chen, Gang and Zhou, Huihui and Tian, Yonghong}, 
        journal={IEEE Transactions on Knowledge and Data Engineering}, 
        title={A Hybrid Spiking Neurons Embedded LSTM Network for Multivariate Time Series Learning under Concept-drift Environment}, 
        year={2022}, 
        pages={1-14}, 
        doi={10.1109/TKDE.2022.3178176}}
@github: https://github.com/zwd2016/HSN-LSTM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class VariableAttention(nn.Module):
    def __init__(self, input_num, n_units, init_std=0.02):
        super(VariableAttention, self).__init__()
        self.weight_w = torch.nn.Parameter(torch.randn(input_num, n_units, 1)*init_std,requires_grad = True)
        self.bias = torch.nn.Parameter(torch.randn(input_num, 1)*init_std,requires_grad = True)

    def forward(self, hidden_states):
        '''
            attn for each time series
            hidden_states : [B, L, N, D]
            queries : [N, Q, D]  Q = 1
        '''
        # calculate scores  [B, L, N, D] * [N, Q, D] ——> [B, L, N, Q]
        alphas = torch.tanh(torch.einsum('ndq,blnd->blnq', self.weight_w,hidden_states) + self.bias)
        alphas = torch.exp(alphas)
        scores = F.softmax(alphas, dim = 1)
        # weighted hidden_states   [B, L, N, Q] * [B, L, N, D] ——> [B, Q, N, D]
        p = torch.einsum("blnq,blnd->bqnd", scores, hidden_states)[:,0,:,:]
        # [B, N, D]
        return p
    

class FusionAttention(nn.Module):
    def __init__(self, out_dim, n_units):
        super(FusionAttention, self).__init__()
        self.func_q = nn.Linear(2 * n_units, out_dim)
        self.func_g = nn.Linear(2 * n_units, 1)

    def forward(self, hidden_state, p):
        '''
            hidden_state : [B, N, D]
            p : [B, N, D]
        '''
        # concate    [B, N , 2 * D]
        new_states = torch.cat((hidden_state, p),dim=2)
        # calculate u  [B, N, out_dim]
        mu = self.func_q(new_states)

        if new_states.size(1) > 1:
            # calculate s         [1, 2 * D],[B, N , 2*D] ——>[B, N ,1]    (torch.einsum("qd,bnd->bnq", v, new_states))
            s = self.func_g(new_states)
            s = torch.exp(s)
            s = s / torch.sum(s,dim=1,keepdim=True)
            s = 1e-2 * -torch.log2(s)

            # weighted hidden_states  [B, out_dim]
            y_pred = torch.sum(s*mu, dim=1)

            return torch.unsqueeze(y_pred,dim=2)
        else:
            return mu.permute(0,2,1)