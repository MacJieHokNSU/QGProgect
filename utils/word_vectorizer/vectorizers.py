from abc import ABCMeta
from abc import abstractmethod
from pathlib import Path
from typing import List

import h5py
import numpy as np
from keras import backend as K
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import Reshape
from keras.layers import TimeDistributed
from keras.models import Model

from .highway_layer import Highway



class WordVectorizer(metaclass=ABCMeta):
    @abstractmethod
    def vectorize(self, words: List[str]) -> List[np.ndarray]: ...


class CharCnnWordVectorizer(WordVectorizer):
    __MAX_TOKEN_LENGTH = 50
    __ALPHABET_SIZE = 261
    __SYMBOL_VECTOR_SIZE = 16

    __SENTENCE_START = 256
    __SENTENCE_END = 257
    __WORD_START = 258
    __WORD_END = 259
    __PADDING = 260

    def __init__(self, weights_file: Path) -> None:
        self.__model = self.__build_model()
        self.__load_weights(self.__model, weights_file)

    def vectorize(self, words: List[str]) -> List[np.ndarray]:
        if len(words) == 0:
            return []

        encoded_sentence = np.array([[
            self.__convert_word_to_embedding_indices(word)
            for word in words
        ]])

        vectorized_sentence = self.__model.predict(encoded_sentence)[0]

        return [vectorized_word.flatten() for vectorized_word in vectorized_sentence]

    @classmethod
    def __convert_word_to_embedding_indices(cls, word: str) -> np.ndarray:
        encoded_word = np.array([int(b) for b in word.encode('utf-8')[:cls.__MAX_TOKEN_LENGTH - 2]])

        embedding_indices = np.array([cls.__PADDING] * cls.__MAX_TOKEN_LENGTH)
        embedding_indices[0] = cls.__WORD_START
        embedding_indices[1:encoded_word.size + 1] = encoded_word
        embedding_indices[encoded_word.size + 1] = cls.__WORD_END

        return embedding_indices

    @classmethod
    def __build_model(cls) -> Model:
        input_layer = Input(
            shape=(None, cls.__MAX_TOKEN_LENGTH),
            name='input'
        )

        embedding_layer = Embedding(
            input_dim=cls.__ALPHABET_SIZE,
            output_dim=cls.__SYMBOL_VECTOR_SIZE,
            name='embedding'
        )(input_layer)

        reshape_layer = TimeDistributed(
            Reshape(input_shape=(cls.__MAX_TOKEN_LENGTH, cls.__SYMBOL_VECTOR_SIZE),
                    target_shape=(1, cls.__MAX_TOKEN_LENGTH, cls.__SYMBOL_VECTOR_SIZE)),
            name='convolution_reshape',
        )(embedding_layer)

        convolutions = []
        filters = (
            (1, 32),
            (2, 32),
            (3, 64),
            (4, 128),
            (5, 256),
            (6, 512),
            (7, 1024),
        )
        for i, (filter_width, n_filters) in enumerate(filters):
            convolution_layer = TimeDistributed(
                Convolution2D(filters=n_filters, kernel_size=(1, filter_width)),
                name=f'convolution_{i}'
            )(reshape_layer)

            max_pooling_layer = TimeDistributed(
                MaxPooling2D(pool_size=(1, cls.__MAX_TOKEN_LENGTH - filter_width + 1)),
                name=f'max_pooling_{i}',
            )(convolution_layer)

            activation_relu_layer = TimeDistributed(
                Activation('relu'),
                name=f'activation_relu_{i}',
            )(max_pooling_layer)

            squeeze_layer = activation_relu_layer
            for j in range(1):
                squeeze_layer = TimeDistributed(
                    Lambda(lambda t: K.squeeze(t, axis=2)),
                    name=f'squeeze_{i}',
                )(activation_relu_layer)

            convolutions.append(squeeze_layer)

        concatenate_layer = Concatenate(name='concat_convolutions')(convolutions)

        highway_layer = concatenate_layer
        for i in range(2):
            highway_layer = Highway(name=f'highway_{i}')(highway_layer)

        projection_layer = Dense(
            512,
            name='projection'
        )(highway_layer)

        return Model(inputs=input_layer, outputs=projection_layer)

    @staticmethod
    def __load_weights(model: Model, weights_file: Path) -> None:
        with h5py.File(weights_file.as_posix(), 'r') as cnn_model_weights_file:
            model.get_layer(name='embedding').set_weights([
                cnn_model_weights_file['char_embed'].value,
            ])

            for i in range(7):
                model.get_layer(name=f'convolution_{i}').set_weights([
                    cnn_model_weights_file['CNN'][f'W_cnn_{i}'].value,
                    cnn_model_weights_file['CNN'][f'b_cnn_{i}'].value,
                ])

            for i in range(2):
                model.get_layer(name=f'highway_{i}').set_weights([
                    cnn_model_weights_file[f'CNN_high_{i}']['W_carry'].value,
                    cnn_model_weights_file[f'CNN_high_{i}']['b_carry'].value,
                    cnn_model_weights_file[f'CNN_high_{i}']['W_transform'].value,
                    cnn_model_weights_file[f'CNN_high_{i}']['b_transform'].value,
                ])

                model.get_layer(name='projection').set_weights([
                    cnn_model_weights_file['CNN_proj']['W_proj'].value,
                    cnn_model_weights_file['CNN_proj']['b_proj'].value,
                ])


__all__ = [
    'WordVectorizer',
    'CharCnnWordVectorizer',
]

