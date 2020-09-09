import argparse
import json
import os


if __name__ == "__main__":
    arguments_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arguments_parser.add_argument('--data_path', help='corrected CoQA ru data file path', type=str, required=False, default="../data/CoQA.json")
    arguments_parser.add_argument('--split_factor', help='split data factor', type=str, required=False, default=0.99)
    args = arguments_parser.parse_args()

    data_path = args.data_path
    data = json.load(open(data_path))

    data_len = len(data)
    train_data = data[:int(data_len*args.split_factor)]
    test_data = data[int(data_len*args.split_factor):]

    train_data_save_name = f"train_{os.path.basename(data_path)}"
    train_data_save_path = os.path.join(os.path.dirname(data_path), train_data_save_name)
    test_data_save_name = f"test_{os.path.basename(data_path)}"
    test_data_save_path = os.path.join(os.path.dirname(data_path), test_data_save_name)

    with open(train_data_save_path, 'w') as fout:
        json.dump(train_data, fout)

    with open(test_data_save_path, 'w') as fout:
        json.dump(test_data, fout)

