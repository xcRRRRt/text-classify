import json
from collections import Counter
from typing import Union, Iterable, Mapping, Sequence

import jieba
from tqdm import tqdm

from module.utils import load_data, load_stop_words


class Tokenizer:
    UNK = '<UNK>'
    PAD = '<PAD>'

    def __init__(
            self,
            pad_size: int,
            wordwise: bool = True,
            vocab_dict: Mapping[str, int] = None,
    ):
        """
        分词器
        :param pad_size: 填充tokens到pad_size大小
        :param wordwise: 是否按照词分割句子
        :param vocab_dict: 词表
        """
        jieba.initialize()
        self.wordwise = wordwise
        self.vocab_dict = vocab_dict
        self.pad_size = pad_size
        self.stopwords = set()

    def _cut_and_filter(self, text) -> list[str]:
        """
        jieba分词及过滤停用词
        :param text: 文本
        :return: 过滤后的分词列表
        """
        if not self.wordwise:
            words = [char for char in text]
        else:
            words = list(jieba.cut(text))
        return list(filter(lambda w: w not in self.stopwords, words))

    def build_vocab(
            self,
            data_path: str,
            stopwords: Sequence[str] = None,
            min_frequency: int = 5,
            high_freq_delete_rate: float = 0.,
            max_vocab_size: int = 16384) -> None:
        """
        构建词表
        :param data_path: 数据集路径(单指本项目)
        :param stopwords: 停用词
        :param min_frequency: 最低纳入词表的词频
        :param high_freq_delete_rate: 过滤出现频率最高的词的比率
        :param max_vocab_size: 词表最大尺寸
        :return:
        """
        assert min_frequency > 0, "min frequency must > 0"
        assert high_freq_delete_rate >= 0, "high freq delete rate must >= 0"
        assert max_vocab_size > 0, "max vocab size must > 0"
        texts, labels = load_data(data_path)
        self.stopwords = set(stopwords) if stopwords else set()
        vocab_dict = Counter()
        for text in tqdm(texts, desc="Building vocabulary"):
            words = self._cut_and_filter(text)
            vocab_dict.update(words)

        vocab_list = sorted([pair for pair in vocab_dict.items() if pair[1] >= min_frequency], key=lambda x: x[1], reverse=True)
        high_freq_split = int(high_freq_delete_rate * len(vocab_list))
        vocab_list = vocab_list[high_freq_split:max_vocab_size + high_freq_split]

        vocab_dict = {pair[0]: idx for idx, pair in enumerate(vocab_list)}
        vocab_dict.update({self.UNK: len(vocab_dict), self.PAD: len(vocab_dict) + 1})
        print("Vocabulary size: ", len(vocab_dict))
        self.vocab_dict = vocab_dict

    def load_vocab_dict(self, vocab_dict_path: str):
        with open(vocab_dict_path, "r", encoding="utf-8") as f:
            self.vocab_dict = json.load(f)

    def encode_text(self, texts: Union[Iterable[str], str]) -> Union[list[int], list[list[int]]]:
        """
        token化文本
        :param texts: 单个文本或多个文本
        :return: 单个文本的tokens或多个文本的tokens列表
        """

        def encode_single_text(text: str) -> list[int]:
            words = self._cut_and_filter(text)
            if len(words) >= self.pad_size:
                words = words[:self.pad_size]
            else:
                words = words + [self.PAD] * (self.pad_size - len(words))
            tokens = [self.vocab_dict.get(word, self.vocab_dict[self.UNK]) for word in words]
            return tokens

        if isinstance(texts, str):
            return encode_single_text(texts)
        elif isinstance(texts, list):
            return [encode_single_text(text) for text in texts]
