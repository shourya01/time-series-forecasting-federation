import torch.nn as nn

class MSELoss(nn.Module):
    
    def __init__(self):
        super(MSELoss,self).__init__()
        self.loss = nn.MSELoss()
        
    def forward(self, target, prediction):
        target0 = target[:,:,0].unsqueeze(-1)
        mean, std = target[:,:,1].unsqueeze(-1), target[:,:,2].unsqueeze(-1)
        target = std*target0 + mean
        return self.loss(prediction,target)