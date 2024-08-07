import torch
from torch.utils.data import Dataset

from module.tokenizer import Tokenizer
from module.utils import load_data


class TextDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: Tokenizer):
        """
        数据集
        :param data_path: 数据集路径(单指本项目)
        :param tokenizer: 分词器
        """
        self.tokenizer = tokenizer
        self.ori_texts, self.ori_labels = load_data(data_path)
        self.labels = set(self.ori_labels)
        self._idx_label_map = {idx: label for idx, label in enumerate(self.labels)}
        self.label_idx_map = {label: idx for idx, label in enumerate(self.labels)}

    @property
    def idx_label_map(self):
        return self._idx_label_map

    @idx_label_map.setter
    def idx_label_map(self, new_v: dict[int, str]):
        self._idx_label_map = new_v
        self.label_idx_map = {label: idx for idx, label in new_v.items()}

    def __getitem__(self, idx):
        text = self.ori_texts[idx]
        label = self.ori_labels[idx]

        tokens = self.tokenizer.encode_text(text)
        return torch.Tensor(tokens).long(), int(self.label_idx_map[label])

    def __len__(self):
        return len(self.ori_labels)
