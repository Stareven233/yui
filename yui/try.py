import numpy as np
import bisect

# import test
# test.test_datasets((0, 44.384))

# test.test_pre_postprocess()

# midi = 'MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--3.midi'
# midi = 'MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--3_processeds100.midi'
# midi = 'MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--3_processeds1000.midi'
# midi = 'MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--3_processeds999.midi'
# test.test_midi_diff(r'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0/2015', midi)


print(np.ceil(np.log10(1000)))
start_time = 3.2514
duration =19.6355
segment_sec = 4.096
time = duration - start_time
segment_num = int(time // segment_sec)
segment_num += int(segment_num*segment_sec < time)
sample_list = []

resume_meta = (5, 7.3474)
start_list = np.arange(start_time, duration, segment_sec)
if resume_meta is not None:
  _, st = resume_meta
  pos = bisect.bisect_left(start_list, st)
  start_list = start_list[pos:]
print(start_list)
segment_num = len(start_list)
id_list = [5] * segment_num
sample_list.extend(list(zip(id_list, start_list)))
print(sample_list)

# class A:
#   def __init__(self):
#     self._a = 214
#     self.b = -214
#     self.__c = 1240

# a = A()
# a._a = 435
# print(a._a)
# print(a.b)
# print(a.__c)

# # Example of target with class indices：target是所预测类别的序号，每个batch对应一个标量
# loss = torch.nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)  # tensor([2, 0, 4])
# print(input, target)
# output = loss(input, target)
# print(output)
# output.backward()
# print(output)

# # Example of target with class probabilities：target.shape = (batch, class)，每个batch行都有class个浮点数，表示每个对应类别的概率，行内和为1
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randn(3, 5).softmax(dim=1)  # tensor([[0.1280, 0.0391, 0.1111, 0.5339, 0.1879], [0.1725, 0.0780, 0.2495, 0.1030, 0.3970], [0.1363, 0.3072, 0.2955, 0.0626, 0.1984]]
# print(input, target)
# output = loss(input, target)
# print(output)
# output.backward()
# print(output)

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
