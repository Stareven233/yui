import logging
import os

import numpy as np
import librosa
import note_seq

from config.data import YuiConfig
import preprocessors
import vocabularies
import event_codec
from utils import get_feature_desc


class MaestroDataset:
  def __init__(
    self, 
    dataset_dir: str, 
    config: YuiConfig, 
    codec: event_codec.Codec,
    vocabulary: vocabularies.GenericTokenVocabulary,
    meta_file: str='maestro-v3.0.0.csv'
  ):
    """This class takes the meta of an audio segment as input, and return 
    the waveform and targets of the audio segment. This class is used by 
    DataLoader. 
    
    Args:
      meta_path: str, the filepath of maestro dataset's metadata
    """

    self.dataset_dir = dataset_dir
    self.config = config
    self.meta_dict = preprocessors.read_metadata(f'{dataset_dir}/{meta_file}')
    self.random_state = np.random.RandomState(config.RANDOM_SEED)
    self.codec = codec
    self.vocabulary = vocabulary


  def __getitem__(self, meta):
    """Prepare input and target of a segment for training.
    
    arg:
      meta: tuple(id, start_time), e.g. (1, 8.192) 
    """
  
    idx, start_time = meta
    audio, midi = self.meta_dict['audio_filename'][idx], self.meta_dict['midi_filename'][idx]
    logging.info(f'get {meta=} of {audio=}')
    audio = os.path.join(self.dataset_dir, audio)
    midi = os.path.join(self.dataset_dir, midi)

    duration = self.meta_dict['duration'][idx]
    # meta中记录的音频实际时长，用于计算后面的frame_times
    end_time = min(start_time+self.config.segment_second, duration)
    # 限制切片不超出总长
    audio, _ = librosa.core.load(audio, sr=self.config.SAMPLE_RATE, offset=start_time, duration=end_time-start_time)
    # 每次只读取所需的切片部分，提速效果显著
    ns = note_seq.midi_file_to_note_sequence(midi)
    logging.info(f'{audio.shape=}')
    # logging.info(repr(ns)[:200])

    f = preprocessors.extract_features2(audio, ns, self.config, self.codec, start_time, end_time, example_id=str(meta))
    # TODO targets(events)每次只计算一部分，但需要整个读取midi

    # f = preprocessors.extract_features(audio, ns, duration, self.config, self.codec, include_ties=False, example_id=str(meta))
    # f = preprocessors.extract_target_sequence_with_indices(f)
    # # inputs, <class 'numpy.ndarray'>, shape=(512, 128); targets, <class 'numpy.ndarray'>, shape=(645,);
    # f = preprocessors.map_midi_programs(f, self.codec)
    # f = preprocessors.run_length_encode_shifts_fn(f, self.codec, key='targets', state_change_event_types=['velocity', 'program'])

    # t1 = f["targets"]
    # t2 = f2["targets"]
    # logging.info(f'{t1=}, {t2=}')
    # logging.info(f'{t1==t2}')

    f = preprocessors.compute_spectrograms(f, self.config)
    f = preprocessors.tokenize(f, self.vocabulary, key='targets', with_eos=True)
    # inputs, <class 'numpy.ndarray'>, shape=(512, 512); targets, <class 'numpy.ndarray'>, shape=(33,);
    f = preprocessors.convert_features(f, self.config)

    f["id"] = str(meta)
    # 以音频id(csv表中的行号)与start_time标识一段样本
    return f


class MaestroSampler:
  """Sampler is used to sample segments for training or evaluation.

  arg:
    meta_path: str
    split: 'train' | 'validation' | 'test'
    batch_size: int

  return:
    list[(id, start_time1), (id, start_time2), ...]
    like: [(1, 0.0), (1, 8.192), (1, 16.384), (1, 24.576)]
  """

  def __init__(self, meta_path:str, split:str, batch_size:int, segment_second:float):
    assert split in {'train', 'validation', 'test', }, f'invalid {split=}'

    self.split = split
    self.meta_dict = preprocessors.read_metadata(meta_path, split=self.split)
    self.audio_num = len(self.meta_dict['split'])
    self.batch_size = batch_size
    # self.random_state = np.random.RandomState(cf.RANDOM_SEED)
    self.segment_sec = segment_second
    # _audio_to_frames之后(5868, 128), 预计取(segment_length=1024, 128)，按sr=16kHz，即8.192s
    self.pos = 0
    # 当前子集中文件索引，从0~audio_num-1

  def __iter__(self):
    # 时长d(s), 采样率sr(Hz), 序列长度len(): len = sr * d
    # 但maestro metadata里写的duration跟实际长度对不上，最后几秒也确实听不出声音

    batch_list = []
    total_segment_num = 0

    while True:
      duration = self.meta_dict['duration'][self.pos]
      segment_num = int(duration//self.segment_sec)
      segment_num += int(segment_num*self.segment_sec < duration)
      # 这首曲子切出来的总段数
      start_list = [round(self.segment_sec*i, 3) for i in range(segment_num)]
      # 减少训练时长起见，先不重叠地切片，TODO 后面改成随机起点允许重叠切片，且max_input_length=1024
      # 比如每次切成3段取中间一段，下次就选择剩下2段中较长的作为起点终点
      # start_list[-1] = duration-self.segment_sec
      # 修剪最后一个起点，防止切片超出；TODO 但这样会造成切片重叠，影响最后的后处理

      id_list = [self.meta_dict['id'][self.pos]] * segment_num
      # csv通过pandas读出来后加入的id，同样是从0开始数第一首歌，表头行不计入
      # 这里示例取1是因为第0首属于validation集，而默认split=train
      batch_list.extend(list(zip(id_list, start_list)))
      # [(1, 0.0), (1, 8.192), (1, 16.384), (1, 24.576)]
      total_segment_num += segment_num

      while total_segment_num >= self.batch_size:
        batch, batch_list = batch_list[:self.batch_size], batch_list[self.batch_size:]
        total_segment_num -= self.batch_size
        yield batch
      # 若不够打包成batch就通过外层while再切一首歌

      self.pos = (self.pos + 1) % self.audio_num

  def __len__(self):
    return 1
    # __len__() should return >= 0
      
  def state_dict(self):
    ...

  def load_state_dict(self, state):
    self.pos = state['pos']
    self.batch_list = state['batch_list']


def collate_fn(data_dict_list):
    """Collate input and target of segments to a mini-batch.

    Args:
      data_dict_list: e.g. [
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
        ...
      ]

    Returns:
      array_dict: e.g. {
        'waveform': (batch_size, segment_samples)
        'frame_roll': (batch_size, segment_frames, classes_num), 
        ...
      }
    """
    
    # key: ['encoder_input_tokens', 'encoder_input_mask', 'decoder_input_tokens', 'decoder_target_tokens', 'decoder_target_mask', 'id']
    array_dict = {}
    for key in data_dict_list[0].keys():
      array_dict[key] = np.asarray([data_dict[key] for data_dict in data_dict_list])
      # 由于target每个batch大小不一，无法在此使用np.array统一为ndarray
      # logging.info(f'{array_dict[key].dtype=}, {array_dict[key].shape=}')
    
    # TODO 用NAMESPACE代替dict
    return array_dict


if __name__ == '__main__':
  from torch.utils.data import DataLoader
  from config.data import YuiConfigDev
  from utils import get_feature_desc

  cf = YuiConfigDev()
  train_sampler = MaestroSampler(os.path.join(cf.DATASET_DIR, 'maestro-v3.0.0_tiny.csv'), 'train', batch_size=1, segment_second=cf.segment_second)
  train_dataset = MaestroDataset(cf.DATASET_DIR, cf, meta_file='maestro-v3.0.0_tiny.csv')
  # inputs.shape=(1024, 128), input_times.shape=(1024,), targets.shape=(8290,), input_event_start_indices.shape=(1024,), input_event_end_indices.shape=(1024,), input_state_event_indices.shape=(1024,), 
  train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=0, pin_memory=True)
  # 程序在运行时启用了多线程，而多线程的使用用到了freeze_support()函数，其在类unix系统上可直接运行，在windows系统中需要跟在main后边
  # 因此不在__name__ == '__main__'内部运行时需要将 num_workers 设置为0，表示仅使用主进程

  # 经过collate_fn处理后各特征多了一维batch_size（将batch_size个dict拼合成一个大dict）
  # inputs.shape=(4,1024,128), input_times.shape=(4,1024,), targets.shape=(4,8290,), input_event_start_indices.shape=(4,1024,), input_event_end_indices.shape=(4,1024,), input_state_event_indices.shape=(4,1024,), 
  it = iter(train_loader)
  f = next(it)
  print(get_feature_desc(f))
