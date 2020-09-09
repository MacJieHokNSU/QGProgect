import argparse
import json
from tqdm import tqdm
from pathlib import Path
from nltk import word_tokenize
from utils.text_cleaners import TextCleaner
from utils.word_vectorizer.vectorizers import CharCnnWordVectorizer

if __name__ == "__main__":
    arguments_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arguments_parser.add_argument('--data_path', help='CoQA ru data file path', type=str, required=False, default="../data/CoQA.json")
    args = arguments_parser.parse_args()

    data_path = args.data_path
    data = json.load(open(data_path))

    cleaner = TextCleaner()

    all_words = []
    for sample in data:
        all_words.extend(word_tokenize(sample['story'].lower()))
        for question in sample["questions"]:
            all_words.extend(word_tokenize(cleaner.clean(question['input_text']).lower()))
        for answer in sample["answers"]:
            all_words.extend(word_tokenize(cleaner.clean(answer['span_text']).lower()))
            all_words.extend(word_tokenize(cleaner.clean(answer['input_text']).lower()))

    all_words = list(set(all_words))
    word_vectorizer = CharCnnWordVectorizer(Path('../utils/word_vectorizer/char_cnn_weights.hdf5'))

    output_filename = 'myemb.512'
    with open(output_filename, 'w') as fout:
        for word in tqdm(all_words):
            fout.write(word)
            fout.write(' ')
            word_vector = word_vectorizer.vectorize([word])[0]
            vector_str = ""
            for x in word_vector:
                vector_str += f"{x :.3f} "
            fout.write(vector_str[:-1])
            fout.write('\n')
