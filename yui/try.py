import numpy as np

from test import test_pre_postprocess, test_dataloader
test_pre_postprocess()

# start_list = np.arange(0, 45.12631, 4.096)
# start_list = np.round(start_list, 3)
# print(start_list)

# # 信号
# s_len = 1024 * 128
# samples = np.cos(2*np.pi*200*np.arange(s_len)/10000)
# print(samples.shape)

# # 求mel spectrogram
# # n_mels为梅尔滤波器的数目

# config = DevConfig()
# spec = librosa.feature.melspectrogram(
#   samples, sr=config.SAMPLE_RATE, n_fft=config.FFT_SIZE, 
#   hop_length=config.HOP_WIDTH, win_length=config.FFT_SIZE,
#   window='hann', center=False, pad_mode='reflect', n_mels=config.NUM_MEL_BINS, 
#   fmin=config.MEL_LO_HZ, fmax=config.MEL_HI_HZ  #, norm=1  # 将三角mel权重除以mel带的宽度（区域归一化）
# ).T

# print(spec)

# # 转为DB单位
# spec = librosa.power_to_db(spec)
# print(spec)


# d = [{'Alice': 2341, 'Beth': 9102, 'Cecil': 3258}, {'a': 1, 'b': 2, 'b': 3}]
# d = {'input': np.random.randint(0, 100, size=(3, 12, 4,)), 'times': np.random.randint(0, 100, size=(3, 12,))}

# def p(l:list[dict[str, int]]) -> tuple[dict[str, int]]:
#   return tuple(l)

# v = [[1, 2, 3, 4], [5], [], [6, 7, 8, 9], [10]]
# v = [[1, 2, 3, 4], [6, 7, 8, 9]]



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
