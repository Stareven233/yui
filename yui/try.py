import numpy as np
import tensorflow as tf
from typing import List, Tuple, Sequence, Mapping


# d = [{'Alice': 2341, 'Beth': 9102, 'Cecil': 3258}, {'a': 1, 'b': 2, 'b': 3}]


# def p(l:list[dict[str, int]]) -> tuple[dict[str, int]]:
#   return tuple(l)

# v = [[1, 2, 3, 4], [5], [], [6, 7, 8, 9], [10]]
# v = [[1, 2, 3, 4], [6, 7, 8, 9]]
v = [1, 2, 3, 4]

v = tf.constant(v)
v = tf.expand_dims(v, 0)
v = tf.RaggedTensor.from_tensor(v)
print(tf.rank(v))
eos_shape = tf.concat([v.bounding_shape()[:-2], [1, 1]], axis=0)
eos_id = tf.broadcast_to(3, eos_shape)
print(eos_id)
last_in_sequence = tf.concat([v[..., -1:, :], eos_id], axis=-1)
v = tf.concat([v[..., :-1, :], last_in_sequence], axis=-2)

# v = np.asarray(v)
# v = np.expand_dims(v, axis=0)
# eos_shape = np.concatenate([v.shape[:-2], [1, 1]], axis=0)
# eos_id = np.full(eos_shape, 3, dtype=int)
# print(eos_id)
# last_in_sequence = np.concatenate([v[..., -1:, :], eos_id], axis=-1)
# print(v[..., -1:, :])
# print()
# v = np.concatenate([v[..., :-1, :], last_in_sequence], axis=-2)
# print(v[..., -1:, :])
# print()
print(v, v.dtype)
