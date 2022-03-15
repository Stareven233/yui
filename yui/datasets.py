import numpy as np
import librosa
import note_seq
from torch.utils.data import DataLoader

from config.data import cf
import preprocessors
import vocabularies
from utils import get_feature_desc


class MaestroDataset:
  def __init__(self, dataset_dir:str):
    """This class takes the meta of an audio segment as input, and return 
    the waveform and targets of the audio segment. This class is used by 
    DataLoader. 
    
    Args:
      meta_path: str, the filepath of maestro dataset's metadata
    """

    self.dataset_dir = dataset_dir
    self.meta_dict = preprocessors.read_metadata(f'{dataset_dir}/maestro-v3.0.0.csv')
    self.random_state = np.random.RandomState(cf.RANDOM_SEED)

  def __getitem__(self, meta):
    """Prepare input and target of a segment for training.
    
    arg:
      meta: tuple(id, start_time), e.g. (5, 65.0) 
    """
  
    idx, start_time = meta
    audio, midi = self.meta_dict['audio_filename'][idx], self.meta_dict['midi_filename'][idx]
    
    audio, _ = librosa.core.load(f'{self.dataset_dir}/{audio}', sr=cf.SAMPLE_RATE, offset=start_time, duration=cf.segment_second)
    ns = note_seq.midi_file_to_note_sequence(f'{self.dataset_dir}/{midi}')
    codec = vocabularies.build_codec(cf)
    # vocabulary = vocabularies.vocabulary_from_codec(codec)

    f = preprocessors.extract_features(audio, ns, cf, include_ties=False, codec=codec, example_id=str(idx))
    print(get_feature_desc(f))

    return f


class MaestroSampler:
  """Sampler is used to sample segments for training or evaluation.

  arg:
    meta_path: str
    split: 'train' | 'validation' | 'test'
    batch_size: int

  return:
    list[(id, start_time1), (id, start_time2), ...]
  """

  def __init__(self, meta_path:str, split:str, batch_size:int):
    assert split in ('train', 'validation', 'test', )

    self.split = split
    self.meta_dict = preprocessors.read_metadata(meta_path, split=self.split)
    self.audio_num = len(self.meta_dict['split'])
    self.batch_size = batch_size
    # self.random_state = np.random.RandomState(cf.RANDOM_SEED)
    self.segment_sec = cf.segment_second
    # _audio_to_frames之后(5868, 128), 预计取(segment_length=1024, 128)，按sr=16kHz，即8.192s
    # TODO 之后看情况改成取(segment_length=512~2000, 128)
    self.pos = 0
    # 当前子集中文件索引，从0~audio_num-1
    self.batch_list = []

  def __iter__(self):
    # 时长d(s), 采样率sr(Hz), 序列长度len(): len = sr * d
    # 但maestro metadata里写的duration跟实际长度对不上，最后几秒也确实听不出声音

    while True:
      duration = self.meta_dict['duration'][self.pos]
      start_list = np.arange(0, duration, self.segment_sec)
      # 减少训练时长起见，先不重叠地切片，TODO 后面改成随机起点允许重叠切片
      # 比如每次切成3段取中间一段，下次就选择剩下2段中较长的作为起点终点
      start_list[-1] = duration-self.segment_sec
      # 修剪最后一个起点，防止切片超出
      batch_num = len(start_list)
      id_list = [self.meta_dict['id'][self.pos]] * batch_num
      self.batch_list.extend(list(zip(id_list, start_list)))

      if batch_num >= self.batch_size:
        batch, self.batch_list = self.batch_list[:self.batch_size], self.batch_list[self.batch_size:]
        yield batch

      self.pos = (self.pos + 1) % self.audio_num

  def __len__(self):
    return 1
    # __len__() should return >= 0
      
  def state_dict(self):
    state = {
      'pos': self.pos, 
      'batch_list': self.batch_list
    }
    return state
          
  def load_state_dict(self, state):
    self.pos = state['pos']
    self.batch_list = state['batch_list']


def collate_fn(list_data_dict):
    """Collate input and target of segments to a mini-batch.

    Args:
      list_data_dict: e.g. [
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
        ...]

    Returns:
      np_data_dict: e.g. {
        'waveform': (batch_size, segment_samples)
        'frame_roll': (batch_size, segment_frames, classes_num), 
        ...}
    """
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict


if __name__ == '__main__':
  train_sampler = MaestroSampler(f'{cf.DATASET_DIR}/maestro-v3.0.0_tiny.csv', 'train', batch_size=4)
  # print(len(train_sampler), next(iter(train_sampler)))
  train_dataset = MaestroDataset(cf.DATASET_DIR)
  # train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=1, pin_memory=True)
