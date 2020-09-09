from typing import Optional
from typing import Tuple

from keras import backend as K
from keras.layers import Layer


class Highway(Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.__carry_kernel = None
        self.__carry_bias = None
        self.__transform_kernel = None
        self.__transform_bias = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        kernel_shape = (input_shape[3], input_shape[3])
        bias_shape = (input_shape[3],)

        self.__carry_kernel = self.add_weight(
            name='carry_kernel',
            shape=kernel_shape,
            trainable=True,
            initializer='identity'
        )

        self.__carry_bias = self.add_weight(
            name='carry_bias',
            shape=bias_shape,
            trainable=True,
            initializer='zeros'
        )

        self.__transform_kernel = self.add_weight(
            name='transform_kernel',
            shape=kernel_shape,
            trainable=True,
            initializer='identity'
        )

        self.__transform_bias = self.add_weight(
            name='transform_bias',
            shape=bias_shape,
            trainable=True,
            initializer='zeros'
        )

        super().build(input_shape)

    def call(self, x: K.tf.Tensor) -> K.tf.Tensor:
        carry_gate = K.sigmoid(K.dot(x, self.__carry_kernel) + self.__carry_bias)
        transform_gate = K.relu(K.dot(x, self.__transform_kernel) + self.__transform_bias)

        return carry_gate * transform_gate + (1.0 - carry_gate) * x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return input_shape


__all__ = [
    'Highway',
]

