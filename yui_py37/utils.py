import os
import logging
from datetime import datetime
import shutil

import numpy as np
import torch
import pydub


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
):
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


def move_to_device(batch, device: torch.device):
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
    return f'type(f)={type(f)}, shape={f.shape}, dtype={f.dtype}; '
  elif not isinstance(f, dict):
    return f'type(f)={type(f)}, str(f)={str(f)}'

  desc = ''
  for k, v in f.items():
    desc += f'{k}, {type(v)}, '
    if isinstance(v, (np.ndarray, torch.Tensor)):
      desc += f'shape={v.shape}, dtype={v.dtype}; ' 
    elif isinstance(v, (list, tuple,)):
      desc += f'len={len(v)}; '
    else:
      desc += f'str(v)={str(v)}'
  return desc


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


def int16_to_float32(x):
  return (x / 32767.).astype(np.float32)
