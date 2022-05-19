import os
import logging
from datetime import datetime
import shutil
import warnings

import numpy as np
import torch
import pydub
import matplotlib.pyplot as plt
import pretty_midi
import six
import mido


pretty_midi.pretty_midi.MAX_TICK = 1e7
# 修改pretty_midi的最大tick数，便于读取某些数据集
# https://github.com/craffel/pretty-midi/issues/112

class UnsoundPM(pretty_midi.PrettyMIDI):
  """对待MIDI事件值不严格的PrettyMIDI
  更改了mido.MidiFile构造函数参数clip为True，使得一些事件值超过127的midi文件能够读取，即便并不正确
  """
  
  def __init__(self, midi_file=None, resolution=220, initial_tempo=120):
    if midi_file is not None:
        if isinstance(midi_file, six.string_types):
          midi_data = mido.MidiFile(filename=midi_file, clip=True)
        else:
          midi_data = mido.MidiFile(file=midi_file, clip=True)

        for track in midi_data.tracks:
          tick = 0
          for event in track:
            event.time += tick
            tick = event.time

        self.resolution = midi_data.ticks_per_beat

        self._load_tempo_changes(midi_data)

        max_tick = max([max([e.time for e in t]) for t in midi_data.tracks]) + 1
        if max_tick > pretty_midi.MAX_TICK:
            raise ValueError(('MIDI file has a largest tick of {}, it is likely corrupt'.format(max_tick)))

        self._update_tick_to_time(max_tick)
        self._load_metadata(midi_data)
        if any(e.type in ('set_tempo', 'key_signature', 'time_signature')
          for track in midi_data.tracks[1:] for e in track):
            warnings.warn(
              "Tempo, Key or Time signature change events found on "
              "non-zero tracks.  This is not a valid type 0 or type 1 "
              "MIDI file.  Tempo, Key or Time Signature may be wrong.",
              RuntimeWarning
            )
        self._load_instruments(midi_data)

    else:
        self.resolution = resolution
        self._tick_scales = [(0, 60.0/(initial_tempo*self.resolution))]
        self.__tick_to_time = [0]
        self.instruments = []
        self.key_signature_changes = []
        self.time_signature_changes = []
        self.lyrics = []


class Namespace:
  def __init__(self, **kwargs):
    for name in kwargs:
      setattr(self, name, kwargs[name])

  def __eq__(self, other):
    if not isinstance(other, Namespace):
      return NotImplemented
    return vars(self) == vars(other)

  def __contains__(self, key):
    return key in self.__dict__


default_ts = Namespace(
  numerator = 4,
  denominator = 4,
)

default_ks = Namespace(
  key = 0,
  mode = 0,
)


def draw_picture(row: int, col: int, f_size: tuple):
  r = row
  c = col
  plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
  plt.rcParams['axes.unicode_minus'] = False
  plt.figure(figsize=f_size)
  # plt.figure(figsize=(c*3, r*1+1))

  def inner(idx, pic, title, axis_on="off", cmap='gray'):
    ax = plt.subplot(r, c, idx)
    ax.set_title(title, fontsize=10)
    ax.axis(axis_on)
    ax.set_xticks([])
    ax.set_yticks([])
    if cmap is None:
      plt.imshow(pic)
    else:
      plt.imshow(pic, cmap=cmap)
  return inner


def save_picture(row: int=1, col: int=1, f_size: tuple=(12, 12)):
  ax = plt.subplot(row, col, 1)
  ax.axis("off")

  def inner(img: np.ndarray, path: str, color_map=None):
    plt.figure(figsize=f_size)
    plt.imshow(img, cmap=color_map)
    plt.xticks([])  #去掉x轴
    plt.yticks([])  #去掉y轴
    plt.axis('off')  #去掉坐标轴
    # 必须在imshow调用之后
    plt.savefig(path)
  return inner


def draw_scatter(row: int, col: int):
  r = row
  c = col
  plt.figure(figsize=(14, 8))
  
  def inner(idx, x, y, fsize=6, color="#ff5050"):
    x_d, x_label = x
    y_d, y_label = y
    
    x_min, x_max = x_d.min()-0.5, x_d.max()+0.5
    y_min, y_max = y_d.min()-0.5, y_d.max()+0.5
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.subplot(r, c, idx)
    plt.xlabel(x_label)  # x轴名称
    plt.ylabel(y_label)  # y轴名称
    plt.grid(True)  # 显示网格线
    plt.scatter(x_d, y_d, s=fsize, c=color)
  return inner


class DummySchedule:
  """换成适用固定学习率的Optimizer又不想换代码就用这个"""
  
  def __init__(self, learning_rate) -> None:
    self.lr = learning_rate
  
  def get_lr(self):
    return [self.lr]


def trunc_logits_by_eos(logits, eos_id):
  """根据logits得到pred，再找出每个sample的pred中第一次出现eos的位置
  并将对应logits中该位置往后的值都设置为 [1, 0, ..., 0] 对应token==pad
  迫使loss将eos往后的都当做无效数据计算
  
  但实际上这样更改loss会造成loss不可导吧，没用
  """

  pad = torch.zeros((logits.shape[-1], ), device=logits.device)
  pad[0] = 1  # 用于替代logits最后一维向量成 [1, 0, ..., 0]
  pred = np.argmax(logits.detach().cpu().numpy(), axis=-1)  # (2, 1024)
  indices = np.where(pred == eos_id)
  last_r_idx = -1
  for r_idx, c_idx in zip(*indices):
    if r_idx == last_r_idx:
      continue
      # 只需要每个sample出现eos的第一个位置
    logits[r_idx, c_idx+1:] = pad
    last_r_idx = r_idx


def show_pred(logits, target, mask):
  mask = mask.detach().cpu().numpy()
  pred = np.argmax(logits.detach().cpu().numpy(), axis=-1).reshape(-1)[mask].tolist()
  target = target.detach().cpu().numpy()[mask].tolist()
  print(f'----------\n{pred}\n{target}\n-----------')


def load_audio_segment(
  path: str,
  format: str='mp3',
  sr: int=22050,
) -> pydub.AudioSegment:
  """读取整首音频为单通道
  返回 pydub.AudioSegment 方便按时间切片
  """

  sound = pydub.AudioSegment.from_file(path, format)
  sound = sound.set_frame_rate(sr).set_channels(1)
  return sound


def slice_audio_segment(
  sound: pydub.AudioSegment,
  offset: float=0,
  duration: float=None,
  dtype: np.dtype=np.float32,
) -> np.ndarray:
  """读取整首音频为单通道
  返回 pydub.AudioSegment 方便按时间切片
  offset, duration 单位都是秒
  由于 AudioSegment 切片单位是毫秒，故乘上1000，且内部会转换为整形
  """
  
  start = offset*1000
  end = None if duration is None else (offset+duration)*1000
  sound = sound[start:end]
  audio = np.asarray(sound.get_array_of_samples(), dtype=dtype)
  audio  /= 1 << (8 * sound.sample_width - 1)
  return audio


def load_mp3_mono(
  path: str, 
  sr: int=22050,
  offset: float=None,
  duration: float=None,
  dtype: np.dtype=np.float32,
) -> tuple[np.ndarray, int]:
  """读取mp3为单通道音频
  行为尽量模拟librosa，但使用pydub，速度上快一些
  """

  sound = pydub.AudioSegment.from_file(path, 'mp3', start_second=offset, duration=duration)
  sound = sound.set_frame_rate(sr).set_channels(1)
  # frame_rate=config.SAMPLE_RATE 对mp3无用
  # convert to mono; == np.mean(audio, axis=1)
  audio = np.asarray(sound.get_array_of_samples(), dtype=dtype)
  audio  /= 1 << (8 * sound.sample_width - 1)
  # 缩放到 [-1, 1] 之间
  return audio, sound.frame_rate
# load_mp3_mono(r'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0/2017/MIDI-Unprocessed_070_PIANO070_MID--AUDIO-split_07-08-17_Piano-e_1-02_wav--2.mp3', 16000)


def show_gpu_info():
  if torch.cuda.is_available():
    print('CUDA GPUs are available')  # 是否有可用的gpu
    print('number of avaliable gpu:', torch.cuda.device_count())  # 有几个可用的gpu
    print('index of current device:', torch.cuda.current_device())  # 可用gpu编号
    print('device capability: %d.%d' % torch.cuda.get_device_capability(device=None))  # 可用gpu算力
    print('device name:', torch.cuda.get_device_name(device=None))  # 可用gpu的名字
  else:
    print('No CUDA GPUs are available')


def move_to_device(batch: dict[str, np.ndarray], device: torch.device):
  data_dtype_map = {
    'encoder_input_tokens': None,
    'encoder_input_mask': None,
    'decoder_input_tokens': torch.int64,
    # decoder_input_ids T5可自动根据labels生成
    'decoder_target_tokens': torch.int64,
    # shape=(batch, target_len); int64 即为 longTensor，t5要求的，不然实际上uint16就够了
    'decoder_target_mask': torch.bool,
  }
  res = []
  for k, v in data_dtype_map.items():
    res.append(torch.as_tensor(batch[k], device=device, dtype=v))
    # 类型转换：torch.longTensor, torch.long(), torch.type(), torch.type_as()
  return res


class EarlyStopping:
  def __init__(self, best_path, resume_path, patience=5, verbose=False, delta=0):
    self.best_checkpoint_path = best_path
    self.resume_checkpoint_path = resume_path
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.stop = False
    self.best_val_loss = np.Inf  # float('inf')
    self.delta = delta

  def __call__(self, val_loss):
    # Early stopping
    if val_loss < self.best_val_loss:
      self.best_val_loss = val_loss
      if os.path.isfile(self.best_checkpoint_path):
        os.remove(self.best_checkpoint_path)
        # loss重新降下，超越之前最佳(而不仅是超越上一次)，将best删除
        # 实际上一般情况下best==resume版本，当best不存在代表resume就是最优
      self.counter = 0
    elif self.counter+1 < self.patience:
      if not os.path.exists(self.best_checkpoint_path):
        shutil.copyfile(self.resume_checkpoint_path, self.best_checkpoint_path)
        # loss大于最佳loss，有过拟合倾向，将之前的resume模型作为最优暂存
      self.counter += 1
    else:
      self.stop = True

  def state_dict(self):
    state = {
      'counter': self.counter,
      'stop': self.stop,
      'best_val_loss': self.best_val_loss,
    }
    return state
  
  def load_state_dict(self, state):
    self.counter = state['counter']
    self.stop = state['stop']
    self.best_val_loss = state['best_val_loss']


def count_parameters(model: torch.nn.Module):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_feature_desc(f):
  if isinstance(f, (np.ndarray, torch.Tensor)):
    return f'{type(f)=}, shape={f.shape}, dtype={f.dtype}; '
  elif not isinstance(f, dict):
    return f'{type(f)=}, {str(f)=}'

  desc = ''
  for k, v in f.items():
    desc += f'{k}, {type(v)}, '
    if isinstance(v, (np.ndarray, torch.Tensor)):
      desc += f'shape={v.shape}, dtype={v.dtype}; ' 
    elif isinstance(v, (list, tuple,)):
      desc += f'len={len(v)}; '
    else:
      desc += f'{str(v)=}'
  return desc


def create_folder(fd):
  if not os.path.exists(fd):
    os.makedirs(fd)


def get_filename(path):
  path = os.path.realpath(path)
  name_ext = path.split('/')[-1]
  name = os.path.splitext(name_ext)[0]
  return name


def create_logging(
  log_dir, 
  name='', 
  filemode='w', 
  level=logging.INFO,
  with_time=True,
  print_console=True
):
  create_folder(log_dir)
  # 用DEBUG调试的话会有许多numpy?的输出

  if with_time:
    name = name + '_' + datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f') 
  log_path = os.path.join(log_dir, f'{name}.log')
  logging.basicConfig(
    level=level,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=log_path,
    filemode=filemode
  )
  # %(funcName)s

  # Print to console
  if print_console:
    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter('%(name)s: %(levelname)-4s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

  return logging


def float32_to_int16(x):
  assert np.max(np.abs(x)) <= 1.
  return (x * 32767.).astype(np.int16)
  # 限定必须是绝对值小于1的浮点数才能转换，小数点16位往后的应该是截断，存在误差


def int16_to_float32(x):
  return (x / 32767.).astype(np.float32)


def z_score(x: np.ndarray, axis=-1):
  x -= np.mean(x, axis=axis)
  x /= np.std(x, axis=axis)
  return x
  # 本以为infer的数据需要标准化，可观察到 train 时，
  # int16_to_float32 处理后的数据绝对值存在大于1的
