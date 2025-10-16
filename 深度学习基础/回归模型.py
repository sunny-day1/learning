import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(6,4)
        self.linear2=nn.Linear(4,2)
        self.linear3=nn.Linear(2,1)
        self.activate1=nn.Sigmoid()
        self.activate2=nn.ReLU()
    def forward(self,x):
        x=self.activate2(self.linear1(x))
        x=self.activate2(self.linear2(x))
        x=self.activate1(self.linear3(x))
        if len(x.size()) > 1:
            return x.squeeze(1)
        else:
            return x

