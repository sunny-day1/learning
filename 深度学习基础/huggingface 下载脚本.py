from torch import nn
from transformers import AutoConfig, pipeline
from transformers import AutoTokenizer
from pathlib import Path
import requests
from tqdm import tqdm  # ← 导入 tqdm
MODEL_NAME = "distilbert-base-uncased"
SAVE_ROOT = Path("../resource/hugging_face")
LOCAL_DIR = SAVE_ROOT / MODEL_NAME  # 简化：直接用 MODEL_NAME 作为文件夹名
BASE_URL = f"https://hf-mirror.com/{MODEL_NAME}/resolve/main"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
print(f"📁 已创建文件夹: {LOCAL_DIR.resolve()}")
def download_file(base_url, filename, save_dir):
    url = f"{base_url}/{filename}"
    save_path = Path(save_dir) / filename
    print(f"📥 正在下载: {filename} ...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, stream=True, timeout=30, verify=False, headers=headers)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(save_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                ncols=80,
                colour='green'
        ) as pbar:
            chunk_size = 65536
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))

        print(f"✅ 成功保存: {save_path}")
    except Exception as e:
        print(f"❌ 下载失败 {filename}: {e}")
files = ["config.json",'vocab.json', "pytorch_model.bin", "vocab.txt", "tokenizer_config.json",'special_tokens_map.json']
for file in files:
    download_file(BASE_URL, file, LOCAL_DIR)
print("\n🎉 下载完成！")



# model_ckpt = "../resource/hugging_face/bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# text = "time flies like an arrow"
# inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
# print(inputs.input_ids)
# config = AutoConfig.from_pretrained(model_ckpt)
# print(config)
# token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
# print(token_emb)
# inputs_embeds = token_emb(inputs.input_ids)
# print(inputs_embeds)
# from transformers import pipeline

# download_model_safe.py