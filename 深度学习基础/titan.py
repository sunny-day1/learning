import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from pytorch学习.作业.回归模型 import Model

df = pd.read_csv('../resource/train.csv', index_col='PassengerId')
df.drop(['Name','Cabin','Embarked','Ticket'],inplace=True,axis=1)
df['Sex'].replace({'male':0,'female':1},inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
Y=df['Survived'].values[1:]
X=df.drop('Survived',axis=1).values[1:,::]
x_train,x_val,y_train,y_val=train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y)



class titanDataset(Dataset):
    def __init__(self,x,y):
        self.X=torch.tensor(x,dtype=torch.float32)
        self.Y=torch.tensor(y,dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, item):
        return self.X[item],self.Y[item]


train_dataset=titanDataset(x_train,y_train)
val_dataset=titanDataset(x_val,y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
model=Model()
loss=torch.nn.BCELoss(size_average=True)
epoch=100
optimizer=torch.optim.Adam(model.parameters(), lr=0.01)

for i in range(epoch):
    train_loss=0
    val_acc=0
    for data in train_loader:
        optimizer.zero_grad()
        x,y=data
        pred=model(x)
        bat_loss = loss(pred, y)
        bat_loss.backward()
        optimizer.step()
        train_loss += bat_loss.detach().item()
    if i%10==0:
        print('此时的训练loss为'+str(train_loss))
        with torch.no_grad():
            correct=0
            for data in val_loader:
                x,y=data
                pred=model(x)
                pred_labels = (pred > 0.5).float()
                correct += (pred_labels == y).sum().item()
        val_acc = correct /len(titanDataset(x_val,y_val))
        print("此时的测试准确度为",str(val_acc))










