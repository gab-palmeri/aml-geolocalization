import torch
import math
from torch import nn

# model to ensemble different networks
class ModelEnsembler(nn.Module):
    def __init__(self, models):
        super(ModelEnsembler, self).__init__()
        self.models = models
        
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        # the output of the model is the mean of the outputs of the models
        return torch.mean(torch.stack(outputs), dim=0)
