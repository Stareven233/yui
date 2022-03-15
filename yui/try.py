import numpy as np
import librosa


a, sr = librosa.load(f'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0/2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--3.wav', sr=16000)
print(len(a), sr , len(a)/sr)

# d = [{'Alice': 2341, 'Beth': 9102, 'Cecil': 3258}, {'a': 1, 'b': 2, 'b': 3}]
# d = {'input': np.random.randint(0, 100, size=(3, 12, 4,)), 'times': np.random.randint(0, 100, size=(3, 12,))}

# def p(l:list[dict[str, int]]) -> tuple[dict[str, int]]:
#   return tuple(l)

# v = [[1, 2, 3, 4], [5], [], [6, 7, 8, 9], [10]]
# v = [[1, 2, 3, 4], [6, 7, 8, 9]]


# v = np.array(v)
# for i in v:
#     i = i[:3]
#     i[2] = 0
# print(v)

# frame_size = 128
# samples_len = 9216
# samples = np.random.randint(0, 100, size=(samples_len,))
# print(f'{frame_size - samples_len % frame_size=}')
# print(samples, len(samples))
# samples = np.pad(samples, [0, frame_size - samples_len % frame_size], mode='constant')
# print(samples, len(samples))

# f1 = librosa.util.frame(samples, 128, 128, axis=0)
# f2 = tf.signal.frame(
#     samples,
#     frame_length=128,
#     frame_step=128,
#     pad_end=False
# )
# print(f'{f1.shape=}, {f2.shape=}')
# print(f'{f1=}')
# print(f'{f2=}')


# v = tf.constant(v)
# v = tf.expand_dims(v, 0)
# v = tf.RaggedTensor.from_tensor(v)
# print(tf.rank(v))
# eos_shape = tf.concat([v.bounding_shape()[:-2], [1, 1]], axis=0)
# eos_id = tf.broadcast_to(3, eos_shape)
# print(eos_id)
# last_in_sequence = tf.concat([v[..., -1:, :], eos_id], axis=-1)
# v = tf.concat([v[..., :-1, :], last_in_sequence], axis=-2)

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
# print(v, v.dtype)
