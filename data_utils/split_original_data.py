import argparse
import json
import os
from typing import Dict


# Максимальная длина единицы разбиения в символах
MAX_CHUNK_LEN = 900000


def get_sample_len(sample: Dict) -> int:
    """Вычисляет полный размер примера в символах
    :param sample: dict пример из CoQA датасета
    :return: int общая длина в символах всего значимого текста из примера
    """
    full_len = 0
    text = sample['story']
    full_len += len(text)
    for question in sample['questions']:
        full_len += len(question['input_text'])
    for answer in sample['answers']:
        full_len += len(answer['input_text'])
        full_len += len(answer['span_text'])
    return full_len

if __name__ == "__main__":
    arguments_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arguments_parser.add_argument('--data_path', help='CoQA data file path', type=str, required=True)
    args = arguments_parser.parse_args()

    data_path = args.data_path
    data = json.load(open(data_path))

    print(f"Читаются данные из {data_path}")

    chunks = []
    last_chunk = []
    last_chunk_len = 0
    for sample in data['data']:
        sample_len = get_sample_len(sample)
        if last_chunk_len + sample_len > MAX_CHUNK_LEN:
            chunks.append(last_chunk)
            last_chunk = [sample]
            last_chunk_len = sample_len
        else:
            last_chunk.append(sample)
            last_chunk_len += sample_len

    print(f"Данные разбиты на {len(chunks)} части")

    data_save_root_path = os.path.join(os.path.dirname(data_path), "splitted_dataset")
    if not os.path.exists(data_save_root_path):
        os.mkdir(data_save_root_path)

    for idx, chunk in enumerate(chunks):
        save_path = os.path.join(data_save_root_path, f"CoQA_train_part_{idx}.json")
        with open(save_path, "w") as fout:
            json.dump(chunk, fout)

    print(f"Разбитые данные сохранены в {data_save_root_path}")

