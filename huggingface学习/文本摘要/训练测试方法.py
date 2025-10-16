import torch
from rouge import Rouge
from tqdm import tqdm

from food_classification.pytorch学习.transformer作业.文本摘要.数据集和模型 import device, max_target_length, tokenizer


def train_loop(dataloader,model,optimizer,lr_scheduler,epoch,total_loss):
    progress_bar=tqdm(range(len(dataloader)))
    progress_bar.set_descrition(f'loss:{0:>7f}')
    finish_batch_num=epoch*len(dataloader)

    model.train()
    for batch, batch_data in enumerate(dataloader):
        batch_data=batch_data.to(device)
        outputs=model(**batch_data)
        loss=outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss+=loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch+1):>7f}')
        progress_bar.update(1)
    return total_loss

rouge=Rouge()

def test_loop(dataloader,model):
    preds,labels=[],[]

    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data=batch_data.to(device)
        with torch.no_grad():
            generated_tokens=model.generate(
                batch_data['input_ids'],
                attention_mask=batch_data['attention_mask'],
                max_length=max_target_length,
                num_beams=4,
                no_repeat_ngram_size=2
            ).cpu().numpy()
        label_tokens=batch_data['labels']

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = torch.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
        preds += [' '.join(pred.strip()) for pred in decoded_preds]
        labels += [' '.join(label.strip()) for label in decoded_labels]
    scores = rouge.get_scores(hyps=preds, refs=labels, avg=True)
    result = {key: value['f'] * 100 for key, value in scores.items()}
    result['avg'] = np.mean(list(result.values()))
    print(
        f"Rouge1: {result['rouge-1']:>0.2f} Rouge2: {result['rouge-2']:>0.2f} RougeL: {result['rouge-l']:>0.2f}\n")
    return result



