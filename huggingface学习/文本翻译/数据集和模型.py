import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sentencepiece
max_dataset_size = 220000
train_set_size = 200000
valid_set_size = 20000
max_length=128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint=r'../../resource/hugging_face/Helsinki-NLP/opus-mt-zh-en'
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
model = model.to(device)
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
class TRANS(Dataset):
    def __init__(self,data_file):
        self.df=self.load_data(data_file)
    def load_data(self,data_file):
        df=pd.read_json(data_file,lines=True,nrows=max_dataset_size)
        return df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        return self.df.iloc[item].to_dict()

data = TRANS(r'D:\360安全浏览器下载\translation2019zh\translation2019zh_train.json')
train_data,valid_data=random_split(data,[train_set_size, valid_set_size])
test_data=TRANS(r'D:\360安全浏览器下载\translation2019zh\translation2019zh_valid.json')
print(train_data[0])

def collote_fn(batch):
    batch_inputs=[item['english'] for item in batch]
    batch_targets=[item['chinese'] for item in batch]
    batch_data=tokenizer(
        text=batch_inputs,
        text_target=batch_targets,
        padding=True,
        max_length = max_length,
        truncation=True,
        return_tensors='pt')
    batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(batch_data.labels)
    end_token_index=(batch_data['labels'] == tokenizer.eos_token_id).int().argmax(dim=1)
    for idx,end_idx in enumerate(end_token_index):
        batch_data['labels'][idx][end_idx+1:]=-100
    return batch_data
train_dataloader=DataLoader(train_data,batch_size=32,shuffle=True,collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=32, shuffle=False, collate_fn=collote_fn)
if __name__ == '__main__':
    batch = next(iter(train_dataloader))
    sentences = ['我叫陈文杰，我是武汉大学电子信息的学生。', '我希望未来能有一份至少30w年薪的开发工作']
    sentences_inputs = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True
    ).to(device)
    sentences_generated_tokens = model.generate(
        **sentences_inputs,
        max_length=128
    )
    sentences_decoded_preds = tokenizer.batch_decode(sentences_generated_tokens, skip_special_tokens=True)
    print(sentences_decoded_preds)
