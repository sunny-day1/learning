from torch.utils.data import DataLoader

from pytorch学习.transformer作业.文本翻译.数据集和模型 import TRANS, model, collote_fn
from pytorch学习.transformer作业.文本翻译.训练测试方法 import test_loop

test_data=TRANS(r'D:\360安全浏览器下载\translation2019zh\translation2019zh_valid.json')
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collote_fn)

test_loop(test_dataloader, model)
