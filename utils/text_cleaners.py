import pymorphy2
import re
from functools import lru_cache


PATTERN = re.compile('\W')


class TextCleaner:
    """
    Чистит и лемматизирует предложения
    """
    def __init__(self):
        self.__analizer = pymorphy2.analyzer.MorphAnalyzer()

    @lru_cache(2000)
    def normal_form(self, word):
        return self.__analizer.normal_forms(word)[0]

    def clean(self, text, normalize=False, join=True):
        if normalize:
            result = [self.normal_form(word) for word in PATTERN.sub(' ', text).split()]
        else:
            result = [word for word in PATTERN.sub(' ', text).split()]
        if join:
            result = ' '.join(result)
        return result
