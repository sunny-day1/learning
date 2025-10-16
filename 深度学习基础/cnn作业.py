from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset=datasets.MNIST(root='../resource/',train=True,download=True,transform=transform)
val_dataset=datasets.MNIST(root='../resource/',train=False,download=True,transform=transform)
train_loader=DataLoader(train_dataset,shuffle=True,batch_size=64)
val_loader=DataLoader(val_dataset,shuffle=False,batch_size=64)

model=nn.Sequential(
    nn.Conv2d(1,4,3,1,1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(4,8,3,1,1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(8,16,3,1,1),
    nn.ReLU(),
    nn.MaxPool2d(2,ceil_mode=True),
    nn.Flatten(start_dim=1),
    nn.Linear(256,128),
    nn.Linear(128,64),
    nn.Linear(64,10),
)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
epoch=50
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.AdamW(model.parameters(),lr=0.01)
plt_train_loss = []
val_loss = []
for i in range(epoch):
    train_loss=0
    for data in tqdm(train_loader, desc=f'Epoch {i + 1}/{epoch}'):
        optimizer.zero_grad()
        x,y=data
        x,y=x.to(device),y.to(device)
        pred=model(x)
        bat_loss = loss(pred, y)
        bat_loss.backward()
        optimizer.step()
        train_loss += bat_loss.detach().item()
    plt_train_loss.append(train_loss / train_loader.__len__())
    if i%10==0:
        print('第%d的训练loss为%.2f'%(i,train_loss/train_loader.__len__()))
        with torch.no_grad():
            correct=0
            for data in val_loader:
                x,y=data
                x, y = x.to(device), y.to(device)
                pred=model(x)
                _,pred_labels=torch.max(pred,dim=1)
                correct += (pred_labels == y).sum().item()
        val_acc =correct/len(val_dataset)
        print("此时的测试准确度为",str(val_acc))
plt.plot(plt_train_loss)
plt.title('训练loss和验证集准确度')
plt.show()