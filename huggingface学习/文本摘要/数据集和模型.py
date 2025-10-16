import torch.cuda
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

max_dataset_size=200000

class LCSTS(Dataset):
    def __init__(self, data_file):
        self.data=self.load_data(data_file)

    def load_data(self,data_file):
        Data={}
        with open(data_file,'rt',encoding='utf-8') as f:
            for idx,line in enumerate(f):
                if idx>=max_dataset_size:
                    break
                items=line.strip().split('!=!')
                Data[idx]={
                    'title':items[0],
                    'content':items[1]
                }
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

train_data = LCSTS('../../resource/summary/data1.tsv')
valid_data = LCSTS('../../resource/summary/data2.tsv')
test_data = LCSTS('../../resource/summary/data3.tsv')
print(f'train set size: {len(train_data)}')
print(f'valid set size: {len(valid_data)}')
print(f'test set size: {len(test_data)}')
print(next(iter(train_data)))
checkpoint = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
device='cuda' if torch.cuda.is_available() else 'cpu'
model=AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
max_input_length=512
max_target_length=64
def collote_fn(batch_samples):
    batch_inputs=[item['content'] for item in batch_samples]
    batch_targets=[item['title'] for item in batch_samples]
    batch_data=tokenizer(
        batch_inputs,
        padding=True,
        max_length=max_input_length,
        truncation=True,
        return_tensors='pt'
    )
    with tokenizer.as_target_tokenizer():
        labels=tokenizer(
            batch_targets,
            padding=True,
            max_length=max_target_length,
            truncation=True,
            return_tensors='pt'
        )['input_ids']
        end_token_index=(labels==tokenizer.eos_token_id).int().argmax(dim=-1)
        for idx,end_idx in enumerate(end_token_index):
            labels[idx][end_idx+1:]=-100
        batch_data['labels']=labels
    return batch_data
train_dataloader=DataLoader(train_data,batch_size=8,shuffle=True,collote_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=8, shuffle=False, collote_fn=collote_fn)
batch = next(iter(train_dataloader))
print(batch.keys())
print('batch shape:', {k: v.shape for k, v in batch.items()})
print(batch)



