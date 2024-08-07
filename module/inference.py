from typing import Union, Literal

import torch

from module.pl_model import TextClassifier
from module.tokenizer import Tokenizer


class TextClassifierInference:
    def __init__(self, ckpt_path: str, tokenizer: Tokenizer):
        """
        文本分类器推理
        :param ckpt_path: checkpoint文件，.ckpt文件
        :param tokenizer: 加载完vocab_dict的分词器

        Usage::

        >>> from module.inference import TextClassifierInference
        >>> from module.tokenizer import Tokenizer
        >>> tokenizer = Tokenizer(pad_size=128)
        >>> tokenizer.load_vocab_dict('output/version_3/vocab_dict.json')
        >>> text_classifier = TextClassifierInference(ckpt_path='output/version_3/checkpoints/epoch=5-step=2928.ckpt', tokenizer=tokenizer)
        >>> result = text_classifier.inference("我24 179 76.4 ，健身一年零两个月了，平常是三分化，周六会去跑一个5km，饮食也有刻意的控制，但是也不会特别控制。我想问一下，在我30岁之前可以练成这样的吗？", type_="label")
        >>> print(result)
        """
        assert tokenizer.vocab_dict, r"Please load vocabulary dict before using TextClassifierInference, Use 'Tokenizer.load_vocab_dict(\'path\\to\\vocab\\dict\')'"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TextClassifier.load_from_checkpoint(ckpt_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    def inference(self, text: str, type_: Literal['idx', "label"] = "label") -> Union[str, int]:
        """
        文本分类器推理
        :param text: 要推理的文本
        :param type_: 推理结果，label为真实标签，idx为标签对应的数字， default: 'label'
        :return: 标签或标签对应的id
        """
        assert type_ in ["idx", "label"], "Please choose either 'idx' or 'label' for classifier output"
        tokens = self.tokenizer.encode_text(text)
        tokens = torch.Tensor(tokens).long().unsqueeze(0).to(self.device)
        with torch.no_grad():
            label = self.model(tokens, type_)
        return label
