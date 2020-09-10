# QGProgect
Проект по генерации вопросов в диалоге на русском языке.

Решается задача Conversational Question Answering (СoQA).

Используется полу-автоматически переведенный на русский язык датасет [СoQA](https://stanfordnlp.github.io/coqa/) ([статья](https://arxiv.org/pdf/1808.07042.pdf)).

Обучается модель [ReDR](https://github.com/ZJULearning/ReDR) ([статья](https://www.aclweb.org/anthology/P19-1203.pdf))


## Как использовать

### 1. Скачать веса моделей
[Веса](https://drive.google.com/file/d/1JgGZmkaGU74mSW-1dhgdkppHEKbS5BnH/view?usp=sharing) модели векторизации слов положить в папку `utils/word_vectorizer`. 

[Чекпоинты](https://drive.google.com/drive/folders/18hNnPiNFUY5HSK5Krj37qUI9xjA2B-hd?usp=sharing) обученных моделей положить в корневой каталог

### 2. Чтобы получить прогнозы вопросов и отчет по BLEU-2 из папки `ReDR/` запустить скрипт

```python generage.py-src ../data/coqa-cqg-src.dev.txt -history ../data/coqa-cqg-history.dev.txt -tgt ../data/coqa-cqg-tgt.dev.txt -replace_unk -model ../model/_step_100000.pt -output pred.txt```

### 3. Для обучения модели 

Объединить набор данных

```python data_utils/concat_ru_data.py```

Разбить набор на обучение и тест

```python data_utils/train_test_split.py```

Подготовить датасет

```python ReDR/cqg_preprocess.py```

```python ReDR/preprocess.py -train_src ../data/coqa-cqg-src.train.txt -train_history ../data/coqa-cqg-history.train.txt -train_ans ../data/coqa-cqg-ans.train.txt -train_tgt ../data/coqa-cqg-tgt.train.txt -valid_src ../data/coqa-cqg-src.dev.txt -valid_history ../data/coqa-cqg-history.dev.txt -valid_ans ../data/coqa-cqg-ans.dev.txt -valid_tgt ../data/coqa-cqg-tgt.dev.txt -save_data ../data/coqa-cqg --share_vocab --dynamic_dict```

Запустить обучение

```python ReDR/train.py -data ../data/coqa-cqg -save_model out_model/ -gpu_ranks 0 --optim adam --share_embeddings -word_vec_size 256 -pre_word_vecs_enc ../data/embeddings.enc.pt -pre_word_vecs_dec ../data/embeddings.dec.pt  --fix_word_vecs_enc --fix_word_vecs_dec```


