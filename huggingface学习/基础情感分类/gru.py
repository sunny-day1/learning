import json
import re
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

df = pd.read_csv('../数据包n.tsv/train.tsv', sep='\t')
test_df = pd.read_csv('../数据包n.tsv/test.tsv', sep='\t')  # 确保路径正确
def clean_text(text):
    text = re.sub(r'\s+', ' ', str(text))  # 转字符串 + 压缩空白
    text = text.strip().lower()
    return text
df['Phrase'] = df['Phrase'].apply(clean_text)
test_df['Phrase'] = test_df['Phrase'].apply(clean_text)

word2idx = {"<PAD>": 0, "<UNK>": 1}
idx2word = {0: "<PAD>", 1: "<UNK>"}
counter = 2
for phrase in df['Phrase']:
    for word in phrase.split():
        if word not in word2idx:
            word2idx[word] = counter
            idx2word[counter] = word
            counter += 1
def text_to_ids(text):
    return [word2idx.get(word, 1) for word in text.split()]
sequences = [torch.tensor(text_to_ids(phrase)) for phrase in df['Phrase']]
labels = torch.tensor(df['Sentiment'].values, dtype=torch.long)
class TestDataset(Dataset):
    def __init__(self, df):
        self.phrases = df['Phrase'].values
    def __len__(self):
        return len(self.phrases)
    def __getitem__(self, idx):
        text = self.phrases[idx]
        ids = text_to_ids(text)
        return torch.tensor(ids, dtype=torch.long)
class movieDataset(Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, item):
        return self.x[item],self.y[item]

class GRUSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, num_classes=5):
        super(GRUSentiment, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, hidden = self.gru(x)
        last_output = output[:, -1, :]  # [B, H]
        logits = self.classifier(last_output)  # [B, 3]
        return logits
BATCH_SIZE = 64
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
LR = 0.001
EPOCHS = 5
MAX_LEN = 128

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = [text[:MAX_LEN] for text in texts]
    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return texts, labels
def test_collate_fn(batch):
    # batch 中只有文本（没有标签）
    texts = [text[:MAX_LEN] for text in batch]
    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    return texts  # 只返回 texts
dataset =movieDataset(sequences, labels)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_dataset = TestDataset(test_df)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=test_collate_fn
)
vocab_size = len(word2idx)
model = GRUSentiment(vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total = 0
    for texts, labels in dataloader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(texts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(dataloader):.4f}, Acc: {acc:.2f}%")
    torch.save(model.state_dict(), 'gru_sentiment_model.pth')
    with open('word2idx.json', 'w', encoding='utf-8') as f:
        json.dump(word2idx, f, ensure_ascii=False, indent=2)
    predictions = []
    with torch.no_grad():
        for texts in test_dataloader:
            texts = texts.to(device)
            logits = model(texts)
            _, preds = torch.max(logits, 1)
            predictions.extend(preds.cpu().numpy())

    test_df['Sentiment'] = predictions
    submission = test_df[['PhraseId', 'Sentiment']]
    submission.to_csv('submission.csv', index=False)

    print("🎉提交文件已保存为：submission.csv")
    print(submission.head())