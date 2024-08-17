import torch.nn as nn

class MSELoss(nn.Module):
    
    def __init__(self, normalization_type='minmax'):
        super(MSELoss,self).__init__()
        self.loss = nn.MSELoss()
        self.normalization_type = normalization_type
        
    def forward(self, target, prediction):
        if self.normalization_type == 'minmax':
            target0 = target[:,:,0].unsqueeze(-1)
            min,max = target[:,:,1].unsqueeze(-1), target[:,:,2].unsqueeze(-1)
            target0 = (target0 - min) / (max-min)
        else:
            if self.normalization_type == 'z':
                target0 = target[:,:,0].unsqueeze(-1)
                mean, std = target[:,:,1].unsqueeze(-1), target[:,:,2].unsqueeze(-1)
                target0 = (target0 - mean) / std
            else:
                raise ValueError('normalize_type must be either of <minmax> or <z>')
        return self.loss(prediction,target0)