import torch
import torch.nn as nn
import torch.nn.functional as F

#creat a Model Class that inherits nn.Module
class Model(nn.Module):
    # Input Layer
    # Hidden Layer1 (Number of Neurons)
    # Hidden Layer2 (Number of Neurons)
    # Output Layer
    def __init__(self, in_layer=4, h1=8, h2=9, out_layer=3):
        super().__init__() # instantiate our nn.Module
        self.fc1 = nn.Linear(in_layer,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,out_layer)

    def forward(self,x):
        x = F.gelu(self.fc1)
        x = F.gelu(self.fc2)
        x = self.out(x)

        return x
torch.manual_seed(41)
model = Model()