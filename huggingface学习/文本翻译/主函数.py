import os
from random import random
import numpy as np
import torch
from transformers import AdamW, get_scheduler
from pytorch学习.transformer作业.文本翻译.数据集和模型 import model, train_dataloader, valid_dataloader
from pytorch学习.transformer作业.文本翻译.训练测试方法 import train_loop, test_loop
def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
learning_rate=2e-5
epoch_num=3
optimizer=AdamW(model.parameters(),lr=learning_rate)
lr_scheduler=get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)
total_loss = 0.
best_bleu = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss)
    valid_bleu = test_loop(valid_dataloader, model)
    print(f"BLEU: {valid_bleu:>0.2f}\n")
    if valid_bleu > best_bleu:
        best_bleu = valid_bleu
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_bleu_{valid_bleu:0.2f}_model_weights.bin')
print("Done!")