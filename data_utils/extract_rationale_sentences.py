import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

from nltk import sent_tokenize, word_tokenize

from utils.sample_correctors import SampleCorrector
from utils.sent_vectorizer import TfidfEmbeddingVectorizer
from utils.text_cleaners import TextCleaner
from utils.word_vectorizer.vectorizers import CharCnnWordVectorizer


if __name__ == "__main__":
    arguments_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arguments_parser.add_argument('--data_path', help='CoQA ru data file path', type=str, required=False, default="../data/CoQA.json")
    args = arguments_parser.parse_args()

    data_path = args.data_path
    data = json.load(open(data_path))

    all_sentences = []
    for sample in data:
        all_sentences.extend(sent_tokenize(sample['story']))

    sent_cleaner = TextCleaner()
    word_vectorizer = CharCnnWordVectorizer(Path('../utils/word_vectorizer/char_cnn_weights.hdf5'))
    sent_vectorizer = TfidfEmbeddingVectorizer(word_vectorizer, word_tokenize)

    sent_vectorizer.fit(all_sentences, None)

    correct_data = []
    for sample in tqdm(data):
        try:
            if sample["lang"] == "ru":
                correct_sample = SampleCorrector(sample, sent_vectorizer, sent_cleaner)
                correct_data.append(correct_sample.get_sample_data())
            else:
                correct_data.append(sample)
        except Exception as e:
            print(e)

    correct_data_save_name = f"correct_{os.path.basename(data_path)}"
    correct_data_save_path = os.path.join(os.path.dirname(data_path), correct_data_save_name)
    with open(correct_data_save_path, 'w') as fout:
        json.dump(correct_data, fout)

