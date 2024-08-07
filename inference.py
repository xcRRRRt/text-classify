"""
Script to inference the text

Author: xcRt

Date: 2024/08/07

Usage:

>>> from module.inference import TextClassifierInference
>>> from module.tokenizer import Tokenizer
>>> tokenizer = Tokenizer(pad_size=128)
>>> tokenizer.load_vocab_dict('output/version_3/vocab_dict.json')
>>> text_classifier = TextClassifierInference(ckpt_path='output/version_3/checkpoints/epoch=5-step=2928.ckpt', tokenizer=tokenizer)
>>> result = text_classifier.inference("我24 179 76.4 ，健身一年零两个月了，平常是三分化，周六会去跑一个5km，饮食也有刻意的控制，但是也不会特别控制。我想问一下，在我30岁之前可以练成这样的吗？", type_="label")
>>> print(result)
"""

from module.inference import TextClassifierInference
from module.tokenizer import Tokenizer

if __name__ == '__main__':
    ckpt_path = 'output/version_3/checkpoints/epoch=5-step=2928.ckpt'
    vocab_dict_path = 'output/version_3/vocab_dict.json'

    tokenizer = Tokenizer(pad_size=128)
    tokenizer.load_vocab_dict(vocab_dict_path)

    text_classifier = TextClassifierInference(
        ckpt_path=ckpt_path,
        tokenizer=tokenizer
    )
    print(text_classifier.inference("我24 179 76.4 ，健身一年零两个月了，平常是三分化，周六会去跑一个5km，饮食也有刻意的控制，但是也不会特别控制。我想问一下，在我30岁之前可以练成这样的吗？", type_="label"))
