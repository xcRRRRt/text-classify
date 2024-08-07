from typing import Literal

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch import optim, nn

from module.model import Model

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题
plt.rcParams['font.size'] = 24

torch.set_float32_matmul_precision("medium")


class TextClassifier(pl.LightningModule):
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            num_layers: int,
            hidden_dim: int,
            num_classes: int,
            lr: float,
            cm: bool,
            idx_to_label: dict[int, str]
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.cm = cm
        self.idx_to_label = idx_to_label

        self.model = Model(vocab_size, embedding_dim, num_layers, hidden_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        self.out = []

    def forward(self, x, type_: Literal["idx", "label"] = "label"):
        logits = self.model(x)
        class_ = int(torch.argmax(logits, dim=1).item())
        if type_ == "idx":
            return class_
        else:
            return self.idx_to_label[class_]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        acc = torch.sum(torch.argmax(logits, dim=1) == y).item() / y.size(0)
        self.log_dict({
            'train/loss': loss,
            "train/acc": acc
        })
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        acc = torch.sum(torch.argmax(logits, dim=1) == y).item() / y.size(0)
        self.log_dict({
            'val/loss': loss,
            "val/acc": acc
        })

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        # acc = torch.sum(torch.argmax(logits, dim=1) == y).item() / y.size(0)
        if self.cm:
            self.out.append({
                "y_hat": torch.argmax(logits, dim=1).cpu().numpy(),
                "y": y.cpu().numpy()
            })

    def on_test_epoch_end(self):
        if self.cm:
            all_pred = np.concatenate([out['y_hat'] for out in self.out])
            all_labels = np.concatenate([out['y'] for out in self.out])

            cm = confusion_matrix(all_labels, all_pred)

            plt.figure(figsize=(10, 10))
            labels = list(self.idx_to_label.values())
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels,
                        yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig('confusion_matrix.png')
            print("save confusion_matrix to 'confusion_matrix.png'")
            self.out.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
