import torch
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
from food_classification.pytorch学习.transformer作业.文本摘要.数据集和模型 import LCSTS, collote_fn, model, device, max_target_length
test_data = LCSTS('../../resource/summary/data3.tsv')
test_dataloader=DataLoader(test_data,batch_size=32,shuffle=False,collate_fn=collote_fn)
model.load_state_dict(torch.load())
model.eval()
with torch.no_grad():
    print('evaluating on test set...')
    sources, preds, labels = [], [], []
    for batch_data in tqdm(test_dataloader):
        batch_data = batch_data.to(device)
        generated_tokens = model.generate(
            batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            max_length=max_target_length,
            num_beams=beam_size,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        decoded_sources = tokenizer.batch_decode(
            batch_data["input_ids"].cpu().numpy(),
            skip_special_tokens=True)
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        for source, pred in zip(decoded_sources, decoded_preds):
            results.append({
                "document": source.strip(),
                "prediction": pred.strip()
            })
        with open('test_data_pred.json', 'wt', encoding='utf-8') as f:
            for exapmle_result in results:
                f.write(json.dumps(exapmle_result, ensure_ascii=False) + '\n')


