import argparse
import os
import json
import youtokentome as yttm
from utils.text_cleaners import TextCleaner


def clean_answer(answer):
    answer["input_text"] = cleaner.clean(answer["input_text"])
    answer["span_text"] = cleaner.clean(answer["span_text"])
    return answer


def clean_question(question):
    question["input_text"] = cleaner.clean(question["input_text"])
    return question


def write_answer(answer):
    print(cleaner.clean(answer["input_text"]), file=fd)
    print(cleaner.clean(answer["span_text"]), file=fd)


def write_question(question):
    print(cleaner.clean(question["input_text"]), file=fd)


def correct(sample):
    bad_a_indexes = []
    bad_q_indexes = []
    bad_turn_indexes = []
    for i,a in enumerate(sample['answers']):
        if 'bad_turn' in a:
            bad_a_indexes.append(i)
            bad_turn_indexes.append(a["turn_id"])
    for i, q in enumerate(sample['questions']):
        if q["turn_id"] in bad_turn_indexes:
            bad_q_indexes.append(i)

    questions = []
    answers = []

    for i, a in enumerate(sample['answers']):
        if i not in bad_a_indexes:
            answers.append(a)
            write_answer(a)
    for i, q in enumerate(sample['questions']):
        if i not in bad_q_indexes:
            questions.append(q)
            write_question(q)

    #if len(answers) != len(sample['answers']) or len(questions) != len(sample['questions']):
    #    print("corrected")

    sample['answers'] = answers
    sample['questions'] = questions

    return sample


def check_for_bad(sample):
    join_story = " ".join(word_tokenize(sample["story"])).lower()
    for answer in sample["answers"]:
        join_text = " ".join(word_tokenize(answer["span_text"])).lower()
        for sentence in black_list:
            if sentence in join_text:
                return False
    for question in sample["questions"]:
        if " ".join(word_tokenize(question["input_text"])).lower() == 'кто рассказывает эту историю ?':
            return False
    for sentence in black_list:
        if sentence in join_story:
            return False
    return True

from nltk.tokenize import word_tokenize

if __name__ == "__main__":
    arguments_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arguments_parser.add_argument('--data_dir', help='CoQA ru data files dir', type=str, required=False, default="../data/splitted_dataset")
    arguments_parser.add_argument('--save_data_dir', help='save CoQA ru data files dir', type=str, required=False, default="../data")
    args = arguments_parser.parse_args()

    black_list = ["`` маленькие '' и `` большие '' - это прозвища для учеников и наставников .",
 'в пятницу он добивался отстранения от власти временного правительства таиланда на фоне растущей политической напряженности после свержения бывшего премьер-министра йинглак чинават .',
 'более 8000 полицейских и сотни добровольцев присоединились к поискам ребенка в течение двух дней .',
 "10 ) - ( орк , удерживаемый как '' жена буша `` армией сопротивления бога . повторное изнасилование предметами",
 'рик санторум и ньют гингрич будут соревноваться',
 'лоуден говорит , что она была очень активна с группами чайной партии в неваде . `` я-избиратель чайной партии , абсолютно .',
 '`` я не за медицинскую реформу обамы']

    data_dir = args.data_dir
    save_data_dir = args.save_data_dir

    filenames = os.listdir(data_dir)

    all_samples = []
    checked = 0
    unique_ids = set()
    cleaner = TextCleaner()

    with open("text_for_train.txt", 'w') as fd:
        for filename in filenames:
                full_path = os.path.join(data_dir, filename)
                with open(full_path, 'r') as fin:
                    chunk_data = json.load(fin)
                    for sample in chunk_data:
                        if 'ru' in os.path.basename(filename):
                            sample["id"] = sample["id"][-5:] + sample["id"][:-5]
                            sample['lang'] = 'ru'
                        else:
                            sample['lang'] = 'eng'
                        sample_id = sample["id"]
                        if sample_id not in unique_ids and check_for_bad(sample):
                            unique_ids.add(sample_id)
                            all_samples.append(correct(sample))
        with open("../data/new_dataset_test.json", 'r') as f:
            new_data = json.load(f)
            for sample in new_data:
                sample = correct(sample)

    print(f"full data size {len(all_samples)} unique samples")


    yttm.BPE.train(data="text_for_train.txt", vocab_size=10000, model="bpe.model")
    save_data_path = os.path.join(save_data_dir, 'CoQA.json')

    with open(save_data_path, 'w') as fout:
        json.dump(all_samples, fout)
