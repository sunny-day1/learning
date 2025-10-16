import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_scheduler
import numpy as np
import random
import os
from tqdm import tqdm
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
seed = 7
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

categories = set()
class PeopleDaily(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f.read().split('\n\n')):
                if not line:
                    continue
                sentence, labels = '', []
                for i, item in enumerate(line.split('\n')):
                    char, tag = item.split(' ')
                    sentence += char
                    if tag.startswith('B'):
                        labels.append([i, i, char, tag[2:]])  # Remove the B- or I-
                        categories.add(tag[2:])
                    elif tag.startswith('I'):
                        labels[-1][1] = i
                        labels[-1][2] += char
                Data[idx] = {
                    'sentence': sentence,
                    'labels': labels}
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
train_data = PeopleDaily('../../resource/china-people-daily-ner-corpus/example.train')
valid_data = PeopleDaily('../../resource/china-people-daily-ner-corpus/example.dev')
test_data = PeopleDaily('../../resource/china-people-daily-ner-corpus/example.test')
id2label = {0:'O'}
for c in list(sorted(categories)):
    id2label[len(id2label)] = f"B-{c}"
    id2label[len(id2label)] = f"I-{c}"
label2id = {v: k for k, v in id2label.items()}

checkpoint = "../resource/hugging_face/bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def collate_fn(batch):
    batch_sentence = [item['sentence'] for item in batch]
    batch_tags = [item['labels'] for item in batch]
    batch_inputs = tokenizer(
        batch_sentence,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    y=[]
    for i,j in enumerate(batch_sentence):
        t=tokenizer(j,truncation=True)
        x=np.array([0 for _ in batch_inputs.input_ids[i]])
        x[0]=-100
        x[len(t.input_ids)-1:]=-100
        for char_start,char_end,_,tags in batch_tags[i]:
            token_start=t.char_to_token(char_start)+1
            token_end=t.char_to_token(char_end)+1
            x[token_start]=label2id[f'B-{tags}']
            x[token_start+1:token_end+1]=label2id[f'I-{tags}']
        y.append(x)
    y_array = np.array(y, dtype=np.int64)
    return batch_inputs,torch.from_numpy(y_array)
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collate_fn)
batch_X, batch_y = next(iter(train_dataloader))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_X = {k: v.to(device) for k, v in batch_X.items()}
batch_y = batch_y.to(device)

print(f'Using {device} device')

class BertForNER(nn.Module):
    def __init__(self, config,t):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, t)

    def forward(self, x):
        bert_output = self.bert(**x)
        sequence_output = bert_output.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

model = BertForNER(checkpoint, len(id2label)).to(device)
outputs = model(batch_X)
print(outputs.shape)


def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch - 1) * len(dataloader)

    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred.permute(0, 2, 1), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss / (finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss
def test_loop(dataloader, model):
    true_labels, true_predictions = [], []

    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            predictions = pred.argmax(dim=-1).cpu().numpy().tolist()
            labels = y.cpu().numpy().tolist()
            true_labels += [[id2label[int(l)] for l in label if l != -100] for label in labels]
            true_predictions += [
                [id2label[int(p)] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
    print(classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2))
    return classification_report(
      true_labels,
      true_predictions,
      mode='strict',
      scheme=IOB2,
      output_dict=True
    )
learning_rate = 1e-5
batch_size = 4
epoch_num = 3
total_loss = 0.
best_f1 = 0.
loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    metrics = test_loop(valid_dataloader, model)
    valid_macro_f1, valid_micro_f1 = metrics['macro avg']['f1-score'], metrics['micro avg']['f1-score']
    valid_f1 = metrics['weighted avg']['f1-score']
    if valid_f1 > best_f1:
        best_f1 = valid_f1
        print('saving new weights...\n')
        torch.save(
            model.state_dict(),
            f'epoch_{t+1}_valid_macrof1_{(100*valid_macro_f1):0.3f}_microf1_{(100*valid_micro_f1):0.3f}_weights.bin'
        )
print("Done!")












