"""
Script to train a text classification

Author: xcRt

Date: 2024/07/20

This script can train a text classification model.

Dependencies:
    - jieba==0.42.1
    - lightning==2.3.3
    - matplotlib==3.7.1
    - numpy==1.23.5
    - scikit_learn==1.2.2
    - seaborn==0.13.2
    - torch==2.3.1+cu121
    - tqdm==4.66.4
"""


from module.tokenizer import Tokenizer

from module.trainer import ClassifierTrainer, ClassifierTrainArgs
from module.utils import load_stop_words

if __name__ == '__main__':
    tokenizer = Tokenizer(pad_size=128, wordwise=True)
    tokenizer.build_vocab(
        "./data.json",
    )

    args = ClassifierTrainArgs(
        data_path="data.json",
        lr=1e-4,
        epochs=150,
        batch_size=32,
        num_workers=4,
        embedding_dim=384,
        num_layers=3,
        hidden_dim=384,
        train_split=0.8,
        valid_split=0.1,
        test_split=0.1,
    )

    trainer = ClassifierTrainer(arg=args, tokenizer=tokenizer)
    trainer.train()
    trainer.test(ckpt_path="output/version_3/checkpoints/epoch=5-step=2928.ckpt")
