import logging
import os
import bisect
import time

import numpy as np
import note_seq
import pydub
import h5py

from config.data import YuiConfig
import preprocessors
import vocabularies
import event_codec
import utils


class MaestroDataset:
  def __init__(
    self, 
    dataset_dir: str, 
    config: YuiConfig, 
    codec: event_codec.Codec,
    vocabulary: vocabularies.Vocabulary,
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
    # self.random_state = np.random.RandomState(config.RANDOM_SEED)
    self.codec = codec
    self.vocabulary = vocabulary


  def __getitem__(self, meta):
    """Prepare input and target of a segment for training.
    
    arg:
      meta: tuple(id, start_time), e.g. (1, 8.192) 
    """
  
    idx, start_time = meta
    audio, midi = self.meta_dict['audio_filename'][idx], self.meta_dict['midi_filename'][idx]
    audio = os.path.join(self.dataset_dir, audio)
    midi = os.path.join(self.dataset_dir, midi)

    duration = self.meta_dict['duration'][idx]
    # meta中记录的音频实际时长，用于计算后面的frame_times
    end_time = min(start_time+self.config.segment_second, duration)
    # 限制切片不超出总长

    st = time.time()
    audio, _ = utils.load_mp3_mono(audio, sr=self.config.SAMPLE_RATE, offset=start_time, duration=end_time-start_time)
    # 每次只读取所需的切片部分，提速效果显著
    logging.debug(f'get {meta=}, {audio.shape=}, {time.time()-st}')

    ns = note_seq.midi_file_to_note_sequence(midi)
    # logging.debug(repr(ns)[:200])

    f = preprocessors.extract_features2(audio, ns, self.config, self.codec, start_time, end_time, example_id=str(meta))
    # targets(events)每次只计算一部分，但需要整个读取midi

    # f = preprocessors.extract_features(audio, ns, duration, self.config, self.codec, include_ties=False, example_id=str(meta))
    # f = preprocessors.extract_target_sequence_with_indices(f)
    # # inputs, <class 'numpy.ndarray'>, shape=(512, 128); targets, <class 'numpy.ndarray'>, shape=(645,);
    # f = preprocessors.map_midi_programs(f, self.codec)
    # f = preprocessors.run_length_encode_shifts_fn(f, self.codec, key='targets', state_change_event_types=['velocity', 'program'])

    # t1 = f["targets"]
    # t2 = f2["targets"]
    # logging.debug(f'{t1=}, {t2=}')
    # logging.debug(f'{t1==t2}')

    f = preprocessors.compute_spectrograms(f, self.config)
    f = preprocessors.tokenize(f, self.vocabulary, key='targets', with_eos=True)
    # inputs, <class 'numpy.ndarray'>, shape=(512, 512); targets, <class 'numpy.ndarray'>, shape=(33,);
    f = preprocessors.convert_features(f, self.config)

    f["id"] = str(meta)
    # 以音频id(csv表中的行号)与start_time标识一段样本
    return f


class MaestroDataset2(MaestroDataset):
  def __init__(
    self, 
    dataset_dir: str, 
    config: YuiConfig, 
    codec: event_codec.Codec,
    vocabulary: vocabularies.Vocabulary,
    meta_file: str='maestro-v3.0.0.csv'
  ):
    super().__init__(dataset_dir, config, codec, vocabulary, meta_file)
    self.data_cache: list[tuple[int, pydub.AudioSegment, note_seq.NoteSequence]] = []  # (-9, None, None)
    # 记录该曲子id以及完整数据; 使用 AudioSegment 方便按时间切片
    # 由于 sampler2 采样是顺序进行的，每首曲子切完片才换下一首，这里记录当前读取的曲子数据，避免重复io浪费时间
    self.max_caches_size = self.config.NUM_WORKERS + 1  # 跟数据读取线程数一致

  def __getitem__(self, meta):
    """Prepare input and target of a segment for training.
    
    arg:
      meta: tuple(id, start_time), e.g. (1, 8.192) 
    """
  
    idx, start_time = meta
    # 这里的idx是整个数据集范围的id
    audio, midi = self.meta_dict['audio_filename'][idx], self.meta_dict['midi_filename'][idx]
    audio = os.path.join(self.dataset_dir, audio)
    midi = os.path.join(self.dataset_dir, midi)
    duration = self.meta_dict['duration'][idx]
    end_time = min(start_time+self.config.segment_second, duration)
    
    # st = time.time()
    flag = True
    while flag:
      if not self._is_in_data_cache(idx):
        # chche未命中
        self._updata_data_cache(idx, audio, midi)
        # logging.info('miss cache')
      audio, ns = self._read_data_cache(idx, start_time, end_time)

      f = preprocessors.extract_features2(audio, ns, self.config, self.codec, start_time, end_time)
      f = preprocessors.compute_spectrograms(f, self.config)
      f = preprocessors.tokenize(f, self.vocabulary, key='targets', with_eos=True)
      try:
        f = preprocessors.convert_features(f, self.config)
        flag = False
      except ValueError as e:
        end_time -= 1
        if end_time < start_time:
          raise e
        logging.warning(f'{idx=}, {e}, retry with {start_time=}, {end_time=}')
    # 靠循环处理 target_len 超出长度的问题，每次超出就将 输入切片缩小 再次处理成target
    # 该方法只能是权宜之计，目的是为了不重新训练模型（开销太大了），训练时发生一两次没什么影响
    # 总之时间有限先训练一波试试，以后再增加 length 重新训练（但一般长度确实很少超过1000的）
    # logging.info(f'get {meta=}, {audio.shape=}, {time.time()-st}')

    f["id"] = str(meta)
    return f

  def _is_in_data_cache(self, idx: int) -> int:
    for data in self.data_cache:
      if data[0] == idx:
        return True
    return False

  def _updata_data_cache(self, idx: int, audio: str, midi: str, format: str='mp3') -> None:
    """将音频及midi数据都作为cache暂存，避免多次io、解析带来的时间开销
    读取整首音频为单通道 pydub.AudioSegment 方便按时间切片
    """

    sound = pydub.AudioSegment.from_file(audio, format)
    sound = sound.set_frame_rate(self.config.SAMPLE_RATE).set_channels(1)
    ns = note_seq.midi_file_to_note_sequence(midi)
    if len(self.data_cache) == self.max_caches_size:
      self.data_cache.pop(0)
      # 遵循先进先出原则
      # logging.info('pop cache')
    self.data_cache.append((idx, sound, ns, ))
    # logging.info(f'cache idx: {[i[0] for i in self.data_cache]}')

  def _read_data_cache(
    self,
    idx: int,
    start_time: float=0,
    end_time: float=None,
    dtype: np.dtype=np.float32,
  ) -> np.ndarray:
    """将cache数据取出，音频取其中一段，midi整个拿出
    start_time, end_time 单位都是秒
    由于 AudioSegment 切片单位是毫秒，故乘上1000，且内部会转换为整形
    """
    # 其实midi也该一起切片才符合逻辑，但不好操作，也不是很有必要

    sound = None
    for data in self.data_cache:
      if data[0] == idx:
        _, sound, ns = data
    if sound is None:
      raise KeyError(f'{idx=} not in cache')
    
    start = start_time*1000
    end = None if end_time is None else end_time*1000
    sound = sound[start:end]
    audio = np.asarray(sound.get_array_of_samples(), dtype=dtype)
    audio  /= 1 << (8 * sound.sample_width - 1)
    return audio, ns


class MaestroDataset3(MaestroDataset):
  """总体跟 MaestroDataset2 一样，只是其中数据均使用HDF5文件存储"""

  def __init__(
    self, 
    dataset_dir: str, 
    config: YuiConfig, 
    codec: event_codec.Codec,
    vocabulary: vocabularies.Vocabulary,
    meta_file: str='maestro-v3.0.0.csv'
  ):
    super().__init__(dataset_dir, config, codec, vocabulary, meta_file=meta_file)
    self.data_cache: list[tuple[int, np.ndarray, note_seq.NoteSequence]] = []  # (-9, None, None)
    self.max_caches_size = self.config.NUM_WORKERS + 1  # 跟数据读取线程数一致

  def __getitem__(self, meta):
  
    idx, start_time = meta
    duration = self.meta_dict['duration'][idx]
    end_time = min(start_time+self.config.segment_second, duration)
    # 实际音频时长会比meta里记载的稍长，但多出来的部分实际上并不存在音符
    filename, year = self.meta_dict['audio_filename'][idx], self.meta_dict['year'][idx]
    h5_path = os.path.join(self.dataset_dir, f'{year}.h5')
    filename = os.path.splitext(filename[5:])[0]  # 去掉前面的年份跟后缀
    
    flag = True
    while flag:
      if not self._is_in_data_cache(idx):
        self._updata_data_cache(idx, h5_path, filename)
      audio, ns = self._read_data_cache(idx, start_time, end_time)

      f = preprocessors.extract_features2(audio, ns, self.config, self.codec, start_time, end_time)
      f = preprocessors.compute_spectrograms(f, self.config)
      f = preprocessors.tokenize(f, self.vocabulary, key='targets', with_eos=True)
      try:
        f = preprocessors.convert_features(f, self.config)
        flag = False
      except ValueError as e:
        end_time -= 1
        if end_time < start_time:
          raise e
        logging.warning(f'{idx=}, {e}, retry with {start_time=}, {end_time=}')

    f["id"] = str(meta)
    return f

  def _is_in_data_cache(self, idx: int) -> int:
    for data in self.data_cache:
      if data[0] == idx:
        return True
    return False

  def _updata_data_cache(self, idx: int, h5_path: str, dset_name: str) -> None:
    with h5py.File(h5_path, "r") as f:
      audio = f[f'{dset_name}/audio'][...]
      audio = utils.int16_to_float32(audio)
      ns = f[f'{dset_name}/midi'][...]
      ns = note_seq.NoteSequence.FromString(ns)
      # logging.info(f'load {h5_path=}')

      if len(self.data_cache) == self.max_caches_size:
        self.data_cache.pop(0)
      self.data_cache.append((idx, audio, ns, ))

  def _read_data_cache(
    self,
    idx: int,
    start_time: float=0,
    end_time: float=0,
  ) -> np.ndarray:
    # 其实midi也该一起切片才符合逻辑，但不好操作，也不是很有必要

    audio = None
    for data in self.data_cache:
      if data[0] == idx:
        _, audio, ns = data
    if audio is None:
      raise KeyError(f'{idx=} not in cache')
    
    start = int(start_time * self.config.SAMPLE_RATE)
    end = int(end_time * self.config.SAMPLE_RATE)
    # !时间（秒）乘上采样率就是波形数组的索引!，ndarray与list一样，end超出部分按最大的返回
    audio = audio[start:end]
    return audio, ns


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
      # 减少训练时长起见，先不重叠地切片
      # 比如每次切成3段取中间一段，下次就选择剩下2段中较长的作为起点终点
      # start_list[-1] = duration-self.segment_sec
      # 修剪最后一个起点，防止切片超出；但这样会造成切片重叠，推断时可坑影响后处理

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
    raise NotImplementedError
    # 将每首曲子切片数提前存放在 self.segment_num 就可以计数，但len也没什么必要
      
  def state_dict(self):
    raise NotImplementedError

  def load_state_dict(self):
    raise NotImplementedError


class MaestroSampler2(MaestroSampler):
  """MaestroSampler的改进版
  1. __iter__不再是无穷的，每次只生成一轮所有样本
  2. 实现每首曲子固定长度、随机起点开始(从0~segment_sec挑选)，连续地切片，
    左侧至起点处丢弃，右侧不足长的保留作一个segment，这样一首曲子切成n个定长、1个不定长片段
  3. 支持中断保存状态与恢复
  4. 可选每epoch产生的最大iteration，但当一个epoch结束还不够iteration就直接停止，
    当起作用时该epoch的最后一个iteration由于数据不足会更短
  """
  # 不定长度的切片不好计算总样本数__len__，而且似乎也不是很必要

  def __init__(
    self, 
    meta_path: str, 
    split: str, 
    batch_size: int,
    config: YuiConfig,
    max_iter_num: int=-1,
    drop_last: bool=False,
    loop=False,
  ):
    super().__init__(meta_path, split, batch_size, config.segment_second)
    self.decimal = int(np.ceil(np.log10(config.STEPS_PER_SECOND)))
    # 决定切片时使用的精度
    self.__slice_start = None
    # (audio_num, ) 为每首曲子指定的切片起点，精确到1ms
    self.__audio_idx_list = np.arange(self.audio_num)
    # 通过self.pos作下标，取一个随机数作为切片的audio_index
    self.max_iter_num = max_iter_num
    self.__epoch = 0
    # 标志数据遍历轮数
    self.__resume_meta = None
    # 标志resume时还未处理的第一个sample
    self.config = config
    self.drop_last = drop_last
    self.loop = loop
    # 此时将不断循环采样数据集，不会随着epoch结束，受 max_iter_num 控制
    self.reset_state()

  def __iter__(self):
    sample_list = []
    total_segment_num = 0
    epoch_finish = False
    iteration_cnt = 0
    logging.debug(f'{self.split}, {self.epoch=}, {self.__slice_start=}, {self.__audio_idx_list=}')

    while True:
      idx = self.__audio_idx_list[self.pos]
      duration = self.meta_dict['duration'][idx]
      start_time = self.__slice_start[idx]
      # time = duration - start_time
      # segment_num = int(time // self.segment_sec)
      # # segment_num += int(start_time != 0)
      # # 0到起点处单独一段; 若考虑这段要改写dataset去支持meta=(id, start, end)的索引方式，故舍去
      # segment_num += int(segment_num*self.segment_sec < time)
      # # 末尾不够切的单独一段
      
      start_list = np.arange(start_time, duration, self.segment_sec)
      start_list = np.round(start_list, decimals=self.decimal)
      # [segment_sec*i + start_time for i in range(segment_num)]
      if self.__resume_meta is not None:
        _, st = self.__resume_meta
        pos = bisect.bisect_left(start_list, st)
        start_list = start_list[pos:]
        self.__resume_meta = None

      segment_num = len(start_list)
      id_list = [self.meta_dict['id'][idx]] * segment_num
      sample_list.extend(list(zip(id_list, start_list)))
      # [(1, 0.0), (1, 8.192), (1, 16.384), (1, 24.576)]
      total_segment_num += segment_num

      while total_segment_num >= self.batch_size:
        batch, sample_list = sample_list[:self.batch_size], sample_list[self.batch_size:]
        total_segment_num -= self.batch_size
        yield batch

        if epoch_finish:
          break
        iteration_cnt += 1
        if iteration_cnt == self.max_iter_num:
          if len(sample_list) > 0:
            self.__resume_meta = sample_list[0]
            # kaggle 第二次训练失败处：当sample_list刚好用完还强行取下一个将处理的sample
          else:
            # 这首曲子刚好全部处理完
            self.pos = (self.pos + 1) % self.audio_num  # 下次处理下一首
            self.__epoch += int(self.pos == 0)  # 太巧辣，还正好是一个epoch结束
            self.__resume_meta = None
          return
      # 若不够打包成batch就通过外层while再切一首歌
      # sampler 操作几乎不花时间

      if epoch_finish:
        break

      self.pos = (self.pos + 1) % self.audio_num
      # self.pos==0 说明处理完最后一个样本，马上从头开始新一轮的循环
      self.__epoch += int(self.pos==0)
      if self.loop:
        continue

      epoch_finish = self.pos==0
      if not epoch_finish:
        continue
      elif total_segment_num==0 or self.drop_last:
        # 刚好结束，或丢弃最后不能组成batch的部分
        break
      else:
        # total_segment_num > 0
        self.pos = np.random.randint(0, self.audio_num)  # [low, high)
        self.__init_slice_start()
        # 此时最后一首曲子还有一些片段没能形成batch，随机挑一首曲子再切片，保证最后一首曲子能用完

  def __init_slice_start(self):
    """每个epoch重新初始化起点，每轮的切片就会不同"""
    self.__slice_start = np.random.uniform(0, self.segment_sec, (self.audio_num, ))
    self.__slice_start = np.round(self.__slice_start, decimals=self.decimal)
  
  @property
  def epoch(self):
    return self.__epoch

  @epoch.setter
  def epoch(self, value):
    if not 0 <= value < self.config.NUM_EPOCHS:
      raise ValueError(f"epoch of sampler must between (0, {self.config.NUM_EPOCHS}), got {value}")
    self.__epoch = value

  def reset_state(self):
    self.__init_slice_start()
    self.pos = 0
    if not self.loop:
      np.random.shuffle(self.__audio_idx_list)
    self.__resume_meta = None

  def state_dict(self):
    state = {
      'pos': self.pos,
      'resume_meta': self.__resume_meta,
      'slice_start': self.__slice_start,
      'audio_idx_list': self.__audio_idx_list,
      'epoch': self.__epoch,
    }
    # pos&audio_idx_list指定曲子，resume_meta指定曲子及中断时刻
    return state

  def load_state_dict(self, state):
    self.pos = state['pos']
    self.__resume_meta = state['resume_meta']
    self.__slice_start = state['slice_start']
    self.__audio_idx_list = state['audio_idx_list']
    self.__epoch = state['epoch']


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
      # logging.debug(f'{array_dict[key].dtype=}, {array_dict[key].shape=}')
    
    return array_dict


if __name__ == '__main__':
  from torch.utils.data import DataLoader
  from config.data import YuiConfigDev

  cf = YuiConfigDev()
  # train_sampler = MaestroSampler(os.path.join(cf.DATASET_DIR, 'maestro-v3.0.0_tiny.csv'), 'train', batch_size=1, segment_second=cf.segment_second)
  # train_dataset = MaestroDataset(cf.DATASET_DIR, cf, meta_file='maestro-v3.0.0_tiny.csv')
  # # inputs.shape=(1024, 128), input_times.shape=(1024,), targets.shape=(8290,), input_event_start_indices.shape=(1024,), input_event_end_indices.shape=(1024,), input_state_event_indices.shape=(1024,), 
  # train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=0, pin_memory=True)
  # # 程序在运行时启用了多线程，而多线程的使用用到了freeze_support()函数，其在类unix系统上可直接运行，在windows系统中需要跟在main后边
  # # 因此不在__name__ == '__main__'内部运行时需要将 num_workers 设置为0，表示仅使用主进程

  # # 经过collate_fn处理后各特征多了一维batch_size（将batch_size个dict拼合成一个大dict）
  # # inputs.shape=(4,1024,128), input_times.shape=(4,1024,), targets.shape=(4,8290,), input_event_start_indices.shape=(4,1024,), input_event_end_indices.shape=(4,1024,), input_state_event_indices.shape=(4,1024,), 
  # it = iter(train_loader)
  # f = next(it)
  # print(get_feature_desc(f))


  codec = vocabularies.build_codec(cf)
  vocabulary = vocabularies.vocabulary_from_codec(codec)
  sampler = MaestroSampler2(
    os.path.join(cf.DATASET_DIR, 'maestro-v3.0.0_tiny.csv'), 
    'validation', 
    batch_size=8,
    config=cf,
    max_iter_num=-1
  )
  dataset = MaestroDataset(cf.DATASET_DIR, cf, codec, vocabulary, meta_file='maestro-v3.0.0_tiny.csv')
  loader = DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0, pin_memory=True)
  # for d in loader:
  #   print(get_feature_desc(d))
  print(sampler.audio_num)
  cnt = 0
  for i in sampler:
    print(i)
    cnt += 1
  print(cnt)
