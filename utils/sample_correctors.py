from nltk import sent_tokenize
import numpy as np
from scipy.spatial.distance import cosine
import re


class SampleCorrector:
    def __init__(self, sample_data, vectorizer, cleaner, offset=100, min_len=10, max_len=20):
        self._sample_data = sample_data
        self.min_len = min_len
        self.max_len = max_len
        self._vectorizer = vectorizer
        self._cleaner = cleaner
        self._offset = offset
        self._sentences = sent_tokenize(sample_data['story'], language='russian')
        self._sentences_spans = [self._get_sentence_span(sentence) for sentence in self._sentences]
        self._sentences_vectors = [self._get_vector(sentence) for sentence in self._sentences]
        self._correct_rationale_spans()

    def get_sample_data(self):
        return self._sample_data

    def _get_sentence_span(self, sentence):
        start_idx = self._sample_data["story"].find(sentence)
        end_idx = start_idx + len(sentence)
        return (start_idx, end_idx)

    def _correct_rationale_spans(self):
        for idx, answer in enumerate(self._sample_data['answers']):
            try:
                rationale = self._find_rationale_sentence(answer)
                answer['span_text'] = rationale
            except:
                answer['span_text'] = answer['span_text']

    def _get_candidates(self, span_start, span_end):
        start = span_start - self._offset
        end = span_end + self._offset
        start_idx = 0
        end_idx = len(self._sentences_spans)
        for idx, (sent_start, sent_end) in enumerate(self._sentences_spans):
            if start <= sent_end:
                start_idx = idx
                break
        for idx, (sent_start, sent_end) in enumerate(self._sentences_spans):
            if end <= sent_start:
                end_idx = idx
                break
        return np.arange(start_idx, end_idx)

    def _get_vector(self, sentence):
        return self._vectorizer.transform(self._cleaner.clean(sentence, normalize=False))

    def _is_in(self, req, text):
        return int(self._cleaner.clean(req.lower()) in self._cleaner.clean(text.lower()))

    @staticmethod
    def _get_dist(v1, v2):
        return 1 - cosine(v1, v2)

    def trim_rationale(self, rationale, answer):
        if len(rationale.split()) > self.max_len:
            rationale = self.find_sub_rationale(rationale, answer)
        return rationale

    def find_sub_rationale(self, rationale, answer):
        rationale_parts = [part for part in re.split(r'[;,\,]', rationale) if len(part)]
        rationale_vector = self._get_vector(answer['span_text'])
        candidates_vectors = [self._get_vector(sent) for sent in rationale_parts]
        distances = [self._get_dist(rationale_vector, candidate_vector) for candidate_vector in candidates_vectors]
        rule_distances = [self._is_in(answer['span_text'], sent) for sent in rationale_parts]
        best_candidate_idx = np.argmax(np.array(distances) + np.array(rule_distances))
        rationale = rationale_parts[best_candidate_idx]
        return rationale

    def _find_rationale_sentence(self, answer):
        if len(answer['span_text'].split()) > self.min_len:
            rationale = answer['span_text']
        else:
            candidates = self._get_candidates(answer['span_start'], answer['span_end'])
            rationale_vector = self._get_vector(answer['span_text'])
            candidates_vectors = [self._sentences_vectors[sent_idx] for sent_idx in candidates]
            distances = [self._get_dist(rationale_vector, candidate_vector) for candidate_vector in candidates_vectors]
            rule_distances = [self._is_in(answer['span_text'], self._sentences[sent_idx]) for sent_idx in candidates]
            best_candidate_idx = np.argmax(np.array(distances) + np.array(rule_distances))
            rationale = self._sentences[candidates[best_candidate_idx]]
        rationale = self.trim_rationale(rationale, answer)
        return rationale
