from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class TfidfEmbeddingVectorizer(object):
    """ Для каждого слова текста пытается найти вектор (иначе вектор нулей)
        вектор текста - усредненный взвешенный вектор всех его слов
        веса idf учатся по статистике слов во всех текстах
    """

    def __init__(self, word_vectorizer, word_tokenizer):
        """ Кнструктор принимает модель w2v и токенизатор слов
        """
        self.__word_vectorizer = word_vectorizer
        self.__word_tokenizer = word_tokenizer
        self.__dim = 512

    def fit(self, X, y):
        """ Принимает список текстов выучивает их idf веса
        """
        tfidf = TfidfVectorizer(analyzer="word")
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.__word_weight = defaultdict(lambda: max_idf,
                                         [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self

    def transform(self, X):
        """ Принимает список текстов и трансформирует каждый текст списка в вектор
        """
        words = self.__word_tokenizer(X)
        weights = np.array([self.__word_weight[w] for w in words])
        if len(weights) == 0:
            weights = np.ones(len(words))
        vecs = np.array(self.__word_vectorizer.vectorize(words))
        return np.mean(weights[:, np.newaxis] * vecs, axis=0)
