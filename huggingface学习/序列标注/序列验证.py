import torch
from seqeval.metrics import classification_report
from tqdm import tqdm
from pytorch学习.transformer作业.序列标注.序列标注任务模型 import model, tokenizer, device, id2label, test_dataloader, test_data
import json
model.load_state_dict(
    torch.load('epoch_3_valid_macrof1_95.878_microf1_96.049_weights.bin', map_location=torch.device('cpu'))
)
model.eval()
with torch.no_grad():
    print('evaluating on test set...')
    true_labels, true_predictions = [], []
    for X, y in tqdm(test_dataloader):
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
    results = []
    print('predicting labels...')
    for s_idx in tqdm(range(len(test_data))):
        example = test_data[s_idx]
        inputs = tokenizer(example['sentence'], truncation=True, return_tensors="pt")
        inputs = inputs.to(device)
        pred = model(inputs)
        probabilities = torch.nn.functional.softmax(pred, dim=-1)[0].cpu().numpy().tolist()
        predictions = pred.argmax(dim=-1)[0].cpu().numpy().tolist()

        pred_label = []
        inputs_with_offsets = tokenizer(example['sentence'], return_offsets_mapping=True)
        tokens = inputs_with_offsets.tokens()
        offsets = inputs_with_offsets["offset_mapping"]

        idx = 0
        while idx < len(predictions):
            pred = predictions[idx]
            label = id2label[pred]
            if label != "O":
                label = label[2:] # Remove the B- or I-
                start, end = offsets[idx]
                all_scores = [probabilities[idx][pred]]
                # Grab all the tokens labeled with I-label
                while (
                    idx + 1 < len(predictions) and
                    id2label[predictions[idx + 1]] == f"I-{label}"
                ):
                    all_scores.append(probabilities[idx + 1][predictions[idx + 1]])
                    _, end = offsets[idx + 1]
                    idx += 1

                score = np.mean(all_scores).item()
                word = example['sentence'][start:end]
                pred_label.append(
                    {
                        "entity_group": label,
                        "score": score,
                        "word": word,
                        "start": start,
                        "end": end,
                    }
                )
            idx += 1
        results.append(
            {
                "sentence": example['sentence'],
                "pred_label": pred_label,
                "true_label": example['labels']
            }
        )
    with open('test_data_pred.json', 'wt', encoding='utf-8') as f:
        for exapmle_result in results:
            f.write(json.dumps(exapmle_result, ensure_ascii=False) + '\n')
