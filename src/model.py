import math
import collections
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Logger
logger = Logger()

def freeze_module_params(m):
    if m is not None:
        for p in m.parameters():
            p.requires_grad = False

def get_activation(activate):
    if activate=='tanh':
        return nn.Tanh()
    elif activate=='relu':
        return nn.ReLU()
    elif activate=='gelu':
        return nn.GELU()
    elif activate=='sigmoid':
        return nn.Sigmoid()
    elif activate=='softmax':
        return nn.Softmax()
    else:
        return None


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MaskedConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int,
                kernel_size: int, stride: int=1, dilation: int=1, groups: int=1,
                bias: bool=True):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                        groups=groups, bias=bias, padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)

class ResnetBlock1d(nn.Module):
    def __init__(self, encoder_dim, hidden_size, kernel_size:int, activate='tanh'):
        super(ResnetBlock1d, self).__init__()
        self.activate = get_activation(activate)
        self.cnn1 = MaskedConv1d(hidden_size, hidden_size, kernel_size)
        self.cnn2 = MaskedConv1d(hidden_size, hidden_size, kernel_size)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        identity = x
        x = self.cnn1(x)
        if self.activate is not None:
            x = self.activate(x)
        x = self.cnn2(x)
        x = identity + x
        if self.activate is not None:
            x = self.activate(x)
        return x

class LengthMaxPool1D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.layer(x))
        x = torch.max(x, dim=1)[0]
        return x

class SiameseMuformerDecoder(nn.Module):
    def __init__(self, vocab_size, encoder_dim, hidden_size, activate='tanh', dropout=0.1, kernel_size=3, layers=1, **unused):
        super(SiameseMuformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.fc_reduce = nn.Linear(self.encoder_dim, self.hidden_size)
        self.motif_net = nn.ModuleList([ResnetBlock1d(self.hidden_size, self.hidden_size, kernel_size, activate=activate) for _ in range(layers)])
        self.pooling = LengthMaxPool1D(in_dim=self.hidden_size, out_dim=self.hidden_size)
        self.dist_motifs = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            get_activation(activate),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )

        self.dist_semantics = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            get_activation(activate),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )

    # x1, x2: (B, T, C)
    def motif_feat(self, x, input_mask):
        for index, layer in enumerate(self.motif_net):
            if index == 0:
                x = layer(x, input_mask)
            else:
                x = layer(x)
        return x
        
    # x1, x2: (B, T, C)
    def distinguisher(self, x1, x2, input_mask1, input_mask2):
        x1 = self.fc_reduce(x1)
        x2 = self.fc_reduce(x2)

        x1_semantics = x1[:, 0, :]
        x2_semantics = x2[:, 0, :]

        x1_motifs = self.motif_feat(x1, input_mask=input_mask1)
        x2_motifs = self.motif_feat(x2, input_mask=input_mask2)

        x1_motifs = self.pooling(x1_motifs * input_mask1)
        x2_motifs = self.pooling(x2_motifs * input_mask2)
        
        s_sem = self.dist_semantics(torch.cat([x1_semantics, x2_semantics], dim=-1))
        s_motif = self.dist_motifs(torch.cat([x1_motifs, x2_motifs], dim=-1))

        return s_sem, s_motif

        # x1, x2: (B, T, C)
    def forward(self, x1, x2, x1_rawseq, x2_rawseq, **unused):
        if 'x1_mask' in unused:
            input_mask1 = unused['x1_mask'][:,:,None].repeat(1, 1, self.hidden_size)
        else:
            input_mask1 = None
        if 'x2_mask' in unused:
            input_mask2 = unused['x2_mask'][:,:,None].repeat(1, 1, self.hidden_size)
        else:
            input_mask2 = None

        return self.distinguisher(x1, x2, input_mask1, input_mask2)

class MonoMuformerDecoder(nn.Module):
    def __init__(self, vocab_size, encoder_dim, hidden_size, activate='tanh', dropout=0.1, kernel_size=3, layers=1, **unused):
        super(MonoMuformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        self.fc_reduce = nn.Linear(self.encoder_dim, self.hidden_size)

        self.motif_net = nn.ModuleList([ResnetBlock1d(self.hidden_size, self.hidden_size, kernel_size, activate=activate) for _ in range(layers)])
        self.pooling = LengthMaxPool1D(in_dim=self.hidden_size, out_dim=self.hidden_size)

        self.dist_motifs = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            get_activation(activate),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )

        self.dist_semantics = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            get_activation(activate),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )

    # x1, x2: (B, T, C)
    def motif_feat(self, x, input_mask):
        for index, layer in enumerate(self.motif_net):
            if index == 0:
                x = layer(x, input_mask)
            else:
                x = layer(x)
        return x
        

    # x1, x2: (B, T, C)
    def forward(self, x1, x2, x1_rawseq, x2_rawseq, **unused):
        if 'x1_mask' in unused:
            input_mask1 = unused['x1_mask'][:,:,None].repeat(1, 1, self.hidden_size)
        else:
            input_mask1 = None
        if 'x2_mask' in unused:
            input_mask2 = unused['x2_mask'][:,:,None].repeat(1, 1, self.hidden_size)
        else:
            input_mask2 = None

        x1 = self.fc_reduce(x1)

        # x1_semantics = (x1[:, 0, :] + x1[:, -1, :]) / 2
        # x1_semantics = x1.mean(dim=1)
        x1_semantics = x1[:, 0, :]

        x1_motifs = self.motif_feat(x1, input_mask=input_mask1)

        x1_motifs = self.pooling(x1_motifs * input_mask1)
        
        s_sem = self.dist_semantics(x1_semantics)
        s_motif = self.dist_motifs(x1_motifs)
        
        return s_sem, s_motif

class Muformer(nn.Module):
    def __init__(self, 
        encoder=None, 
        vocab_size=-1, 
        encoder_dim=768, 
        hidden_size=256, 
        num_heads=16, 
        freeze_lm=False,
        biased_layers=None,
        dropout=0.1,
        encoder_name='pmlm',
        decoder_name='mono',
        activate='tanh',
        kernel_size=3,
        conv_layers=1):

        super(Muformer, self).__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.biased_layers = biased_layers
        self.encoder_name = encoder_name
        self.kernel_size = kernel_size
        
        logger.info(f'Hidden size: \t [{self.hidden_size}]')
        logger.info(f'Dropout: \t\t [{self.dropout}]')
        if freeze_lm:
            logger.info('Freezing LM encoder')
            freeze_module_params(self.encoder) ### Freeze the pre-trained LM model
                
        self.attn_map = nn.Sequential(
            nn.Linear(1, 1, bias=False),
            nn.GELU(),
            nn.Linear(1, num_heads, bias=False),
        ) # keep this for reproduction
        
        self.decoder_dict = {
            'siamese': SiameseMuformerDecoder,
            'mono': MonoMuformerDecoder,
        }
        logger.info(f'Decoder: \t\t [{decoder_name}]')
        self.decoder = self.decoder_dict[decoder_name](vocab_size, encoder_dim, hidden_size, dropout=dropout, activate=activate, kernel_size=kernel_size, layers=conv_layers)
        logger.info(f'Vocab size: \t [{vocab_size}]')

        self.mono = (decoder_name == 'mono')

    def diff_prob(self, prob_lm1, prob_lm2, x1_indices, x2_indices):
        prob_lm1.clamp_(min=1e-16)
        logits_lm1 = torch.log_softmax(prob_lm1, dim=-1)
        probs1 = torch.gather(logits_lm1, -1, x1_indices.unsqueeze(-1)).squeeze(-1)

        if self.mono:
            return probs1[:, 1:-1].mean(-1, keepdim=True)
        else:
            prob_lm2.clamp_(min=1e-16)
            logits_lm2 = torch.log_softmax(prob_lm2, dim=-1)
            probs2 = torch.gather(logits_lm2, -1, x2_indices.unsqueeze(-1)).squeeze(-1)
            return (probs1[:, 1:-1].mean(-1, keepdim=True) - probs2[:, 1:-1].mean(-1, keepdim=True)) 
    
    def forward(self, x1, x2, **unused):

        x1_rawseq, x2_rawseq = x1, x2
        x1_indices, x2_indices = x1, x2

        lm_prob1, lm_prob2 = None, None

        x1, _ = self.encoder(x1)
        x1 = x1.permute(1, 0, 2)

        if self.mono:
            x2 = None
        else:
            x2, _ = self.encoder(x2)
            x2 = x2.permute(1, 0, 2)
        
        s_sem, s_motif = self.decoder(x1, x2, x1_rawseq, x2_rawseq, **unused)

        lm_prob1 = self.encoder.prob(x1)
        if self.mono:
            lm_prob2 = None
        else:
            lm_prob2 = self.encoder.prob(x2)
        
        s_prob = self.diff_prob(lm_prob1, lm_prob2, x1_indices, x2_indices)

        output = (s_sem + s_motif + s_prob) / 2

        return output
