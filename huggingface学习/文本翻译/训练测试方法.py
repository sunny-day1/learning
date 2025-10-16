import torch
from sacrebleu import BLEU
from tqdm import tqdm

from pytorch学习.transformer作业.文本翻译.数据集和模型 import device, max_length, tokenizer

bleu=BLEU()
def train_loop(dataloader,model,optimizer,lr_scheduler,epoch,total_loss):
    progress_bar=tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss:{0:>7f}')
    finish_batch_num=epoch*len(dataloader)

    model.train()
    for batch,batch_data in enumerate(dataloader):
        batch_data=batch_data.to(device)
        outputs=model(**batch_data)
        loss=outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss+=loss.item()
        progress_bar.set_description(f'loss:{total_loss/(finish_batch_num+batch+1):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader,model):
    preds,labels=[],[]
    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data=batch_data.to(device)
        with torch.no_grad():
            generated_tokens=model.generate(
                batch_data['input_ids'],
                attention_mask=batch_data['attention_mask'],
                max_length=max_length
            )
            decoded_preds=tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
            label_tokens = torch.where(
                batch_data["labels"] != -100,
                batch_data["labels"],
                tokenizer.pad_token_id)
            decoded_labels=tokenizer.batch_decode(label_tokens,skip_special_tokens=True)
            preds += [pred.strip() for pred in decoded_preds]
            labels += [[label.strip()] for label in decoded_labels]
    return bleu.corpus_score(preds,labels).score




