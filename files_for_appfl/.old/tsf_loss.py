import torch
import torch.nn as nn
import torch.nn.functional as F

class TSFLoss(nn.Module):
    '''Losses for regression based time-series forecasting'''
    def __init__(self, lossfn = F.mse_loss, reduction = 'mean'):
        super(TSFLoss, self).__init__()
        self.criterion = lossfn
        self.reduction = reduction

    def forward(self, prediction, target):
        point_target, full_target = target[0], target[1]
        if prediction.shape == point_target.shape:
            return self.criterion(prediction,point_target,reduction=self.reduction)
        else:
            if prediction.shape == full_target.shape:
                return self.criterion(prediction,full_target,reduction=self.reduction)
            else:
                raise ValueError("Shape of model output does not correspond to either point target or full target.")