from dataclasses import dataclass
import torch
import torch.nn as nn
from models import GPTConfig,transformer_cls

class covered_model(nn.Module):
    def __init__(self, config: GPTConfig):
        super(covered_model, self).__init__()
        self.model = transformer_cls(config)
        self.config = config
        self.device =  'cpu'
        self.to(self.device)

    def forward(self, data):
        if self.config.apply_ehr:
            seq = data[:,:-3].long()
            ehr = data[:,-3:]
            return self.model(seq,ehr)
        else:
            seq = data.long()
            return self.model(seq,torch.zeros((1,3)))
