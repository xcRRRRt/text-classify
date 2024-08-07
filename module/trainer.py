import json
import os
from pathlib import Path
from typing import Union

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import random_split, DataLoader

from module.dataset import TextDataset
from module.pl_model import TextClassifier
from module.tokenizer import Tokenizer


class ClassifierTrainArgs:
    def __init__(
            self,
            data_path: str,

            lr: float,
            epochs: int,

            batch_size: int,
            num_workers: int,

            embedding_dim: int,
            num_layers: int,
            hidden_dim: int,

            train_split: float,
            valid_split: float,
            test_split: float = 0.,

            accelerator="gpu",

            confusion_matrix: bool = True
    ):
        """
        训练参数
        :param data_path: 数据路径(单指本项目那个张磊用∠填的json文件)
        :param lr: learning rate
        :param epochs: epochs
        :param batch_size: batch size
        :param num_workers: number of data loading
        :param embedding_dim: 词向量的维度
        :param num_layers: GRU隐藏层数量
        :param hidden_dim: GRU隐藏层维度
        :param train_split: 训练集占总数据的百分比(0-1), train + valid (+ test) = 1
        :param valid_split: 验证集占总数据的百分比(0-1), train + valid (+ test) = 1
        :param test_split: 测试集占总数据的百分比(0-1), train + valid (+ test) = 1
        :param accelerator: 'gpu' or 'cpu'
        :param confusion_matrix: 是否绘制混淆矩阵
        """
        assert train_split + valid_split + test_split == 1, "train split + valid split (+ test split) = 1"
        self.data_path = data_path
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.valid_split = valid_split
        self.test_split = test_split
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.accelerator = accelerator
        self.confusion_matrix = confusion_matrix


class ClassifierTrainer:
    def __init__(
            self,
            arg: ClassifierTrainArgs,
            tokenizer: Tokenizer
    ):
        """
        训练器
        :param arg: 训练参数
        :param tokenizer: 分词器
        """
        self.dataset = TextDataset(
            data_path=arg.data_path,
            tokenizer=tokenizer,
        )
        self.model = None
        self.arg = arg
        self.logger = TensorBoardLogger("", "output")
        os.makedirs(self.logger.log_dir, exist_ok=True)
        self._save_vocab_dict()
        # self.save_label_idx_map()

    def _save_vocab_dict(self):
        with open(os.path.join(self.logger.log_dir, "vocab_dict.json"), 'w', encoding="utf-8") as file:
            json.dump(self.dataset.tokenizer.vocab_dict, file, ensure_ascii=False, indent=4)

    def _save_label_idx_map(self):
        with open(os.path.join(self.logger.log_dir, "label_idx_map.json"), 'w', encoding='utf-8') as file:
            json.dump({
                "label_idx_map": self.dataset.label_idx_map,
                "idx_label_map": self.dataset.idx_label_map
            }, file, ensure_ascii=False, indent=4)

    def train(self, ckpt_path: Union[str, Path, None] = None):
        trainer = pl.Trainer(
            logger=self.logger,
            accelerator=self.arg.accelerator,
            log_every_n_steps=1,
            max_epochs=self.arg.epochs,
            callbacks=[
                EarlyStopping("val/acc", mode="max", patience=5, verbose=True),
                ModelCheckpoint(monitor="val/acc", mode="max", verbose=True)
            ]
        )
        if ckpt_path:
            self._load_model(ckpt_path)
        else:
            self.model = TextClassifier(
                vocab_size=len(self.dataset.tokenizer.vocab_dict),
                embedding_dim=self.arg.embedding_dim,
                num_layers=self.arg.num_layers,
                hidden_dim=self.arg.hidden_dim,
                num_classes=len(self.dataset.labels),
                lr=self.arg.lr,
                cm=self.arg.confusion_matrix,
                idx_to_label=self.dataset.idx_label_map
            )

        train_set, valid_set, test_set = random_split(dataset=self.dataset, lengths=[self.arg.train_split, self.arg.valid_split, self.arg.test_split])
        train_loader = DataLoader(train_set, batch_size=self.arg.batch_size, shuffle=True, num_workers=self.arg.num_workers, persistent_workers=(self.arg.num_workers > 0))
        valid_loader = DataLoader(valid_set, batch_size=self.arg.batch_size, shuffle=False, num_workers=self.arg.num_workers, persistent_workers=(self.arg.num_workers > 0))
        test_loader = None
        if self.arg.test_split:
            test_loader = DataLoader(test_set, batch_size=self.arg.batch_size, shuffle=False, num_workers=self.arg.num_workers, persistent_workers=(self.arg.num_workers > 0))

        trainer.fit(self.model, train_loader, valid_loader)
        if self.arg.test_split:
            trainer.test(self.model, test_loader)

    def test(self, ckpt_path: Union[str, Path]):
        self._load_model(ckpt_path)
        trainer = pl.Trainer(accelerator=self.arg.accelerator, logger=self.logger)
        _, _, test_set = random_split(dataset=self.dataset, lengths=[self.arg.train_split, self.arg.valid_split, self.arg.test_split])
        test_loader = DataLoader(test_set, batch_size=self.arg.batch_size, shuffle=False, num_workers=self.arg.num_workers, persistent_workers=(self.arg.num_workers > 0))
        trainer.test(self.model, test_loader)

    def _load_model(self, ckpt_path: Union[str, Path]):
        ckpt = torch.load(ckpt_path)
        self.model = TextClassifier.load_from_checkpoint(ckpt_path)
        self.dataset.idx_label_map = ckpt['hyper_parameters']['idx_to_label']
