import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def pearson_loss(x, y):
	mean_x = torch.mean(x)
	mean_y = torch.mean(y)
	xm = x.sub(mean_x)
	ym = y.sub(mean_y)
	r_num = xm.dot(ym)
	r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
	r_val = r_num / r_den
	return 1 - r_val

def pearson_correlation_loss(y_pred, y_true, normalized=False):
    """
     Calculate pearson correlation loss   
    :param y_true: distance matrix tensor tensor size (batch_size, batch_size)
    :param y_pred: distance matrix tensor tensor size (batch_size, batch_size)
    :param normalized: if True, Softmax is applied to the distance matrix
    :return: loss tensor
    """
    if normalized:
        y_true = F.softmax(y_true, axis=-1)
        y_pred = F.softmax(y_pred, axis=-1)

    sum_true = torch.sum(y_true)
    sum2_true = torch.sum(torch.pow(y_true, 2))                         # square ~= np.pow(a,2)

    sum_pred = torch.sum(y_pred)
    sum2_pred = torch.sum(torch.pow(y_pred, 2))

    prod = torch.sum(y_true * y_pred)
    n = y_true.shape[0]                                                     # n == y_true.shape[0]

    corr = n * prod - sum_true * sum_pred

    corr /= torch.sqrt(n * sum2_true - sum_true * sum_true + torch.finfo(torch.float32).eps)
    corr /= torch.sqrt(torch.clamp((n * sum2_pred - sum_pred * sum_pred + torch.finfo(torch.float32).eps), min=0.000001))

    return 1 - corr

### Reference: https://github.com/technicolor-research/sodeep/
class SpearmanLoss(nn.Module):
    """ Loss function  inspired by spearmann correlation.self
    Required the trained model to have a good initlization.

    Set beta to 1 for a few epoch to help with the initialization.
    """
    def __init__(self, beta=0.0):
        super(SpearmanLoss, self).__init__()

        self.criterion_rank = nn.MSELoss()
        self.criterion_score = nn.L1Loss()
        # self.criterion_score = nn.MSELoss()

        self.beta = beta
    
    def get_rank(self, batch_score, dim=0):
        rank = torch.argsort(batch_score, dim=dim)
        rank = torch.argsort(rank, dim=dim)
        rank = (rank * -1) + batch_score.size(dim)
        rank = rank.float()
        rank = rank / batch_score.size(dim)
        return rank
    
    def comp(self, inpu):
        in_mat1 = torch.triu(inpu.repeat(inpu.size(0), 1), diagonal=1)
        in_mat2 = torch.triu(inpu.repeat(inpu.size(0), 1).t(), diagonal=1)

        comp_first = (in_mat1 - in_mat2)
        comp_second = (in_mat2 - in_mat1)

        std1 = torch.std(comp_first).item()
        std2 = torch.std(comp_second).item()

        std1 = torch.finfo(torch.float32).eps if np.isnan(std1) else std1
        std2 = torch.finfo(torch.float32).eps if np.isnan(std2) else std2

        comp_first = torch.sigmoid(comp_first * (6.8 / (std1 + torch.finfo(torch.float32).eps)))
        comp_second = torch.sigmoid(comp_second * (6.8 / (std2 + torch.finfo(torch.float32).eps)))

        comp_first = torch.triu(comp_first, diagonal=1)
        comp_second = torch.triu(comp_second, diagonal=1)

        return (torch.sum(comp_first, 1) + torch.sum(comp_second, 0) + 1) / inpu.size(0)

    def sort(self, input_):
        out = [self.comp(input_[d]) for d in range(input_.size(0))]
        out = torch.stack(out)

        return out.view(input_.size(0), -1)

    def forward(self, mem_pred, mem_gt):
        rank_gt = self.get_rank(mem_gt)

        rank_pred = self.sort(mem_pred.unsqueeze(0)).view(-1)

        return self.criterion_rank(rank_pred, rank_gt) + self.beta * self.criterion_score(mem_pred, mem_gt)

CRITERION = {
    'mae': nn.L1Loss(),
    'mse': nn.MSELoss(),
    'pearson': pearson_correlation_loss,
    'spearman': SpearmanLoss(),
}

def get_criterion(name):
    if name in CRITERION:
        return CRITERION[name]
    else:
        return nn.L1Loss()
