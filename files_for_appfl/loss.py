import torch.nn as nn

class MSELoss(nn.Module):
    
    def __init__(self):
        super(MSELoss,self).__init__()
        self.loss = nn.MSELoss()
        
    def forward(self, prediction, target):
        return self.loss(prediction,target)