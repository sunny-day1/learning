from rouge import Rouge

generated_summary = "我在苏州大学学习计算机，苏州大学很美丽。"
reference_summary = "我在环境优美的苏州大学学习计算机。"

rouge = Rouge()

TOKENIZE_CHINESE = lambda x: ' '.join(x)

# from transformers import AutoTokenizer
# model_checkpoint = "csebuetnlp/mT5_multilingual_XLSum"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# TOKENIZE_CHINESE = lambda x: ' '.join(
#     tokenizer.convert_ids_to_tokens(tokenizer(x).input_ids, skip_special_tokens=True)
# )

scores = rouge.get_scores(
    hyps=[TOKENIZE_CHINESE(generated_summary)],
    refs=[TOKENIZE_CHINESE(reference_summary)]
)[0]
print('ROUGE:', scores)
scores = rouge.get_scores(
    hyps=[generated_summary],
    refs=[reference_summary]
)[0]
print('wrong ROUGE:', scores)
