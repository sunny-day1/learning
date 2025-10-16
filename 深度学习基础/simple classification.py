import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import warnings
import random
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

set_seed()
df=pd.read_csv('../resource/分类train.csv', index_col='id')
df['target']=df['target'].str.replace('Class_','').astype(int)-1
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
x_train,x_val,y_train,y_val=train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
class simDataset(Dataset):
    def __init__(self,x,y):
        self.x=torch.tensor(x,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, item):
        return self.x[item],self.y[item]
train_dataset=simDataset(x_train,y_train)
val_dataset=simDataset(x_val,y_val)
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=20)
epoch=50
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.Sequential(
    nn.Linear(93, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 9)
)
model.to(device)
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

plt_train_loss = []
val_rel = []
for i in range(epoch):
    train_loss=0
    val_acc=0
    for data in train_loader:
        optimizer.zero_grad()
        x,y=data
        x,y=x.to(device),y.to(device)
        pred=model(x)
        bat_loss = loss(pred, y)
        bat_loss.backward()
        optimizer.step()
        train_loss += bat_loss.detach().item()
    plt_train_loss.append(train_loss / train_dataset.__len__())
    if i%10==0:
        print('第%d的训练loss为%.2f'%(i,train_loss/train_dataset.__len__()))
        with torch.no_grad():
            correct=0
            for data in val_loader:
                x,y=data
                x, y = x.to(device), y.to(device)
                pred=model(x)
                _,pred_labels=torch.max(pred,dim=1)
                correct += (pred_labels == y).sum().item()
        val_acc =correct/len(val_dataset)
        val_rel.append(val_acc)
        print("此时的测试准确度为",str(val_acc))

plt.plot(plt_train_loss)
plt.plot(val_rel)
plt.legend(['train_loss', 'val_acc'])
plt.title('训练loss和验证集准确度')
plt.show()


