import random
import numpy as np
import torch
from pytorch学习.作业.回归模型 import Model


def set_deterministic_seed(seed=42):
    # 设置 Python 内置 random
    random.seed(seed)
    # 设置 NumPy
    np.random.seed(seed)
    # 设置 PyTorch CPU 和 GPU
    torch.manual_seed(seed)
set_deterministic_seed(42)

sj=np.loadtxt('../resource/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data=torch.from_numpy(sj[:,:-1])
x_mean = x_data.mean(axis=0)
x_std = x_data.std(axis=0)
x_data = (x_data - x_mean) / x_std
y_data=torch.from_numpy(sj[:,-1:])
model=Model()
loss=torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(100):
    y_pred=model(x_data)
    Loss=loss(y_pred,y_data)
    optimizer.zero_grad()
    Loss.backward()
    optimizer.step()
    print("当下的loss值为"+str(Loss.item()))






