import numpy as np
from config.data import YuiConfigPro
import utils


config = YuiConfigPro(
  DATASET_DIR=r'D:/A日常/大学/毕业设计/dataset/maestro-v2.0.0/',
  DATAMETA_NAME=r'maestro-v3.0.0_tiny.csv',
  WORKSPACE=r'D:/A日常/大学/毕业设计/code/yui/',
  NUM_MEL_BINS=256,
)


from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import torch
from config import build_t5_config

t5_config = build_t5_config(
  d_model=config.NUM_MEL_BINS,
  vocab_size=34000,
  max_length=config.MAX_TARGETS_LENGTH,
)
t5_config = T5Config.from_dict(t5_config)
model = T5ForConditionalGeneration(config=t5_config)
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# the following 2 hyperparameters are task-specific
max_source_length = 512
max_target_length = 128

# Suppose we have the following 2 training examples:
input_sequence_1 = "Welcome to NYC"
output_sequence_1 = "Bienvenue à NYC"

input_sequence_2 = "HuggingFace is a company"
output_sequence_2 = "HuggingFace est une entreprise"

# encode the inputs
task_prefix = "translate English to French: "
input_sequences = [input_sequence_1, input_sequence_2]

encoding = tokenizer(
  [task_prefix + sequence for sequence in input_sequences],
  padding="longest",
  max_length=max_source_length,
  truncation=True,
  return_tensors="pt",
)

input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

# encode the targets
target_encoding = tokenizer(
    [output_sequence_1, output_sequence_2], padding="longest", max_length=max_target_length, truncation=True
)
labels = target_encoding.input_ids

print(encoding, target_encoding)
# {
#   'input_ids': tensor([
#     [13959,  1566,    12,  2379,    10,  5242,    12, 13465,     1,     0,0,     0,     0,     0],
#     [13959,  1566,    12,  2379,    10, 11560,  3896,   371,  3302,    19,3,     9,   349,     1]
#   ]), 
#   'attention_mask': tensor([
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#   ])
# } 
# {
#   'input_ids': [[10520, 15098, 3, 85, 13465, 1, 0, 0], [11560, 3896, 371, 3302, 259, 245, 11089, 1]], 
#   'attention_mask': [[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]]
# }



# replace padding token id's of the labels by -100 so it's ignored by the loss
labels = torch.tensor(labels)
labels[labels == tokenizer.pad_token_id] = -100

# forward pass
loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
print(loss.item())


# audio_name = [
#   # 'MIDI-UNPROCESSED_04-07-08-10-12-15-17_R2_2014_MID--AUDIO_12_R2_2014_wav',
#   'MIDI-UNPROCESSED_21-22_R1_2014_MID--AUDIO_21_R1_2014_wav--2',
# ]
# f = h5py.File(os.path.join(config.DATASET_DIR, f'2004.h5'), "r")
# filepath = '2013/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_06_R1_2013_wav--3.h5'
# f = h5py.File(os.path.join(r'D:\A日常\大学\毕业设计\dataset\hdf5s-v3.0.0', filepath), "r")
# f = h5py.File(f'{config.DATASET_DIR}/hdf5_test.h5', "w")
# origin_audio = None
# for name in audio_name:
#   ns = note_seq.midi_file_to_note_sequence(f'{config.DATASET_DIR}/2014/{name}.midi')
#   # serial = ns.SerializeToString()  # <class 'bytes'> 26071 for 6.34kb
#   # new_ns = note_seq.NoteSequence.FromString(serial)

#   origin_audio, _ = utils.load_mp3_mono(f'{config.DATASET_DIR}/2014/{name}.mp3', sr=config.SAMPLE_RATE)

#   audio_group = f.create_group(name)
#   audio_group.create_dataset('audio', data=utils.float32_to_int16(origin_audio))
#   audio_group.create_dataset('midi', data=np.void(ns.SerializeToString()))

# print(f.keys())
# audio = utils.int16_to_float32(f['waveform'][:])
# print(len(audio))
# print(audio.dtype)
# f.close()

# ns = note_seq.midi_file_to_note_sequence(r'D:\A日常\大学\毕业设计\dataset\maestro-v3.0.0\2013\ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_06_R1_2013_wav--3.midi')
# print(ns.notes[300:])
# audio = f['MIDI-UNPROCESSED_21-22_R1_2014_MID--AUDIO_21_R1_2014_wav--2/audio'][...]
# print(audio.dtype)
# audio = utils.int16_to_float32(audio)
# print(audio.shape)
# print(audio == origin_audio)
# # print(f['MIDI-UNPROCESSED_21-22_R1_2014_MID--AUDIO_21_R1_2014_wav--2/midi'][...])
# f.close()
  


# v = torch.randint(0, 100, (2, 4, 6))
# pad = torch.zeros((6))
# pad[0] = 1
# v = torch.as_tensor(v)
# print(type(v), v)
# v[1, 2:] = pad
# print(type(v), v)

# audio_len = 60408
# num_frames = audio_len // config.FRAME_SIZE
# num_frames += int(audio_len - num_frames*config.FRAME_SIZE > 3)
# print(num_frames)

# audio1 = config.DATASET_DIR + r'2013/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_06_R1_2013_wav--3.mp3'
# audio2 = config.DATASET_DIR + r'2013/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_06_R1_2013_wav--3.wav'
# # audio1 = config.DATASET_DIR + r'2008/MIDI-Unprocessed_11_R2_2008_01-05_ORIG_MID--AUDIO_11_R2_2008_wav--2.mp3'
# # audio2 = config.DATASET_DIR + r'2008/MIDI-Unprocessed_11_R2_2008_01-05_ORIG_MID--AUDIO_11_R2_2008_wav--2.wav'

# audio, _ = utils.load_mp3_mono(audio1, config.SAMPLE_RATE, 3.04, config.segment_second-2.1)
# print(audio.shape, config.segment_second)

# frame_size = config.FRAME_SIZE
# audio_len = len(audio)
# num_frames = np.ceil(audio_len / frame_size).astype(np.int32)
# samples = np.pad(audio, (0, num_frames*frame_size - audio_len), mode='constant')
# # 在末尾补0，便于下面切片；本能整除则不变
# print(f'Padded {audio_len} samples to multiple of {frame_size}')
# frames = librosa.util.frame(samples, frame_length=config.FRAME_SIZE, hop_length=config.HOP_WIDTH, axis=0).astype(np.float32)
# print(f'librosa.util.frame: {frames.shape=}')
# print(f'Encoded {audio_len} samples to {num_frames} frames, {frame_size} samples each')

# mel_spec = librosa.feature.melspectrogram(
#   y=samples, sr=config.SAMPLE_RATE, n_fft=config.FFT_SIZE, 
#   hop_length=config.HOP_WIDTH, win_length=config.FFT_SIZE,
#   window='hann', center=True, pad_mode='reflect', n_mels=config.NUM_MEL_BINS, 
#   fmin=config.MEL_LO_HZ, fmax=config.MEL_HI_HZ
# )
# log_mel_spec = librosa.power_to_db(mel_spec)
# print(f'spectrograms: {log_mel_spec.shape=}')
# log_mel_spec = log_mel_spec.T
# print(f'spectrograms: {log_mel_spec.shape=}')

# def max_length_for_key(key):
#   max_length = getattr(config, f"MAX_{key.upper()}_LENGTH", -1)
#   return max_length

# v =  log_mel_spec
# k = 'inputs'
# v_len = v.shape[0]
# if v_len > (max_v_len := max_length_for_key(k)):
#   print(f'{v_len=} for "{k}" field exceeds maximum length {max_v_len}')
#   exit(-1)

# mask = np.ones((max_v_len, ), dtype=np.bool8)
# v = np.pad(v, [(0, max_v_len-v_len)] + [(0, 0)]*(v.ndim - 1), mode='constant', constant_values=config.PAD_ID)
# mask[v_len:] = 0

# print(v.shape, mask.shape, max_v_len-v_len)


# st = time.time()
# audio, _ = librosa.core.load(audio1, sr=config.SAMPLE_RATE, offset=3.04, duration=4.096)
# print(time.time() - st, audio.shape, audio[:50])
# 2.7461953163146973; 0.616558313369751; 1.2299487590789795

# st = time.time()
# audio, _ = librosa.core.load(audio2, sr=config.SAMPLE_RATE, offset=3.04, duration=4.096)
# print(time.time() - st, audio.shape)
# # 0.2634613513946533; 0.5930068492889404; 0.10272479057312012
# # 放在MP3前面读读取时间会变长，然后mp3变短

# st = time.time()
# sound = pydub.AudioSegment.from_file(audio2, 'wav', frame_rate=config.SAMPLE_RATE, start_second=3.04, duration=4.096)
# print(time.time() - st)
# 0.019203901290893555; 0.0070149898529052734

# st = time.time()
# audio, _ = utils.load_mp3_mono(audio1, config.SAMPLE_RATE, 3.04, 4.096)
# print(time.time() - st, audio.shape, audio[:50])
# # 2.8467371463775635; 0.7002110481262207; 0.6837112903594971


# samples = np.random.rand(131072)
# samples = audio
# print(f'samples: {samples.shape=}')
# mel_spec = librosa.feature.melspectrogram(
#   y=samples, sr=config.SAMPLE_RATE, n_fft=config.FFT_SIZE, 
#   hop_length=config.HOP_WIDTH, win_length=config.FFT_SIZE,
#   window='hann', center=True, pad_mode='reflect', n_mels=config.NUM_MEL_BINS, 
#   fmin=config.MEL_LO_HZ, fmax=config.MEL_HI_HZ
# )
# log_mel_spec = librosa.power_to_db(mel_spec)  # log_mel_spec.shape=(512, 1025)
# print(f'spectrograms: {log_mel_spec.shape=}')
# log_mel_spec = log_mel_spec.T[:512]
# print(f'spectrograms: {log_mel_spec.shape=}')

# a = [False, False, True, False, False, True, False, ]
# print(np.argmax(a))

# a = torch.randint(0, 1<<16, (2, 8, 6))
# print(type(a))
# pred = a.numpy()
# print(type(pred))
# print(pred.shape, pred)
# pred = np.argmax(pred, axis=-1)
# print(pred.shape, pred)

# print(np.ceil(np.log10(1000)))
# start_time = 3.2514
# duration =19.6355
# segment_sec = 4.096
# time = duration - start_time
# segment_num = int(time // segment_sec)
# segment_num += int(segment_num*segment_sec < time)
# sample_list = []

# resume_meta = (5, 7.3474)
# start_list = np.arange(start_time, duration, segment_sec)
# if resume_meta is not None:
#   _, st = resume_meta
#   pos = bisect.bisect_left(start_list, st)
#   start_list = start_list[pos:]
# print(start_list)
# segment_num = len(start_list)
# id_list = [5] * segment_num
# sample_list.extend(list(zip(id_list, start_list)))
# print(sample_list)

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
# d1, d2 = d
# print(d1 | d2 | {'fuck': 421, 'Alice': 'sb'})
# d = {'input': np.random.randint(0, 100, size=(3, 12, 4,)), 'times': np.random.randint(0, 100, size=(3, 12,))}

# def p(l:list[dict[str, int]]) -> tuple[dict[str, int]]:
#   return tuple(l)

# v = [[1, 2, 3, 4], [5], [], [6, 7, 8, 9], [10]]
# v = [[1, 2, 3, 4], [6, 7, 3, 3]]

# v = np.asarray(v)
# r_idx, c_idx = np.where(v==31233)
# for i, d in enumerate(zip(r_idx, c_idx)):
#   print(i, d)

# for row in v:
#   row = np.array([0, 0, 1])
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
