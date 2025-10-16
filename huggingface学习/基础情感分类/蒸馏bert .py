import json
import os
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, set_seed, \
    AutoModel

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
SEED = 42
set_seed(SEED)
cache_dir = './imdb_cache'
dataset = load_dataset('imdb', cache_dir=cache_dir)
train_dataset=dataset['train'].shuffle(seed=SEED).select(range(1000))
test_dataset=dataset['test'].shuffle(seed=SEED).select(range(200))
print(len(dataset['train']))
checkpoint='../../resource/hugging_face/distilbert-base-uncased'
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
def preprocess_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding=True,
        max_length=128
    )
tokenized_train=train_dataset.map(preprocess_function,batched=True)
tokenized_test=test_dataset.map(preprocess_function,batched=True)
tokenized_train=tokenized_train.rename_column('label','labels')
tokenized_test=tokenized_test.rename_column('label','labels')
tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
class CustomSequenceClassification(nn.Module):
    def __init__(self, checkpoint, num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(checkpoint,attention_probs_dropout_prob=0.2)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {'loss': loss, 'logits': logits}
def compute_metrics(eval_pred):
    logits,labels=eval_pred
    predictions=np.argmax(logits,axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {
        'accuracy':acc,
        'f1':f1
    }
for lr in [1e-4, 1e-5, 1e-6]:
    for batch_size in [16, 32]:
        print(f"\n=== Running with lr={lr}, batch_size={batch_size} ===")
        model = CustomSequenceClassification(checkpoint,2)
        training_args=TrainingArguments(
            output_dir=f'./imdb-bert-lr{lr}-bs{batch_size}',
            num_train_epochs=4,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir=f'./logs-lr{lr}-bs{batch_size}',
            logging_steps=60,
            metric_for_best_model="eval_accuracy",
            load_best_model_at_end=True,
            learning_rate=lr,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=60,
            eval_steps=60,
            save_total_limit = 1
        )

        trainer=Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer
        )
        print("开始训练...")
        trainer.train()
        print("开始评估...")
        eval_results = trainer.evaluate()
        print(f"模型修改后的评估结果: lr={lr}, batchsize={batch_size} 结果为 {eval_results}")

        # 保存结果（路径也按参数区分）
        result_path = f"./imdb-bert-lr{lr}-bs{batch_size}/eval_results.json"
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(eval_results, f, indent=4)