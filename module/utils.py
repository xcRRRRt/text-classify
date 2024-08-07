import json
import os


def load_data(data_path) -> tuple[list[str], list[str]]:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [str(item['text']) for item in data]
    labels = [str(item['label']) for item in data]
    return texts, labels


def load_stop_words(stopword_dir: str) -> set[str]:
    stop_words = set()
    if stopword_dir is None:
        return stop_words
    for file in os.listdir(stopword_dir):
        file_path = os.path.join(stopword_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            stop_words.update(f.readlines())
    return stop_words


def is_file_in_directory(file_path, directory):
    # 获取文件的绝对路径
    file_path = os.path.abspath(file_path)
    # 获取目录的绝对路径
    directory = os.path.abspath(directory)
    # 检查文件是否存在于指定目录中
    assert os.path.commonpath([file_path, directory]) == directory, "ckpt_path must be in {}".format(directory)
