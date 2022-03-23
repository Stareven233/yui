import dataclasses

import pretty_midi
pretty_midi.pretty_midi.MAX_TICK = 1e7
# 修改pretty_midi的最大tick数，便于读取某些数据集
# https://github.com/craffel/pretty-midi/issues/112


@dataclasses.dataclass(frozen=True)
class BaseConfig:
  DATASET_DIR: str
  WORKSPACE: str
  TRAIN_STEPS: int
  DEVICE: str

  # io
  RANDOM_SEED:int = 233
  MAX_INPUTS_LENGTH:int =  512  # 指input的第一维，第二维是frame_size: (512, 128)
  MAX_TARGETS_LENGTH:int = 1024  # target第1维: (1024, )
  MAX_SEGMENT_LENGTH:int =  2000
  PROGRAM_GRANULARITY = 'flat'

  # spectrogram
  SAMPLE_RATE:int = 16000
  FRAME_SIZE:int = 128
  HOP_WIDTH:int = FRAME_SIZE
  NUM_MEL_BINS:int = 512
  FFT_SIZE:int = 2048  # fft_window_size and hann_window_size
  MEL_LO_HZ:float = 20.0
  MEL_HI_HZ:float = SAMPLE_RATE / 2
  # 对应 librosa.filters.mel 的fmin跟fmax

  # vocabulary
  ENCODED_EOS_ID:int = 1
  ENCODED_UNK_ID:int = 2
  EXTRA_IDS:int = 100
  DECODED_EOS_ID = -1
  DECODED_INVALID_ID = -2
  STEPS_PER_SECOND = 100
  MAX_SHIFT_SECONDS = 10
  NUM_VELOCITY_BINS = 127

  @property
  def frames_per_second(self):
    return self.SAMPLE_RATE // self.FRAME_SIZE

  @property
  def segment_second(self):
    return (cf.MAX_INPUTS_LENGTH * cf.FRAME_SIZE) / cf.SAMPLE_RATE
  # 以MAX_INPUTS_LENGTH作为一段长度，若小于该值则之后进行pad


@dataclasses.dataclass(frozen=True)
class DevConfig(BaseConfig):
  # io
  DATASET_DIR:str = r'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0/'
  WORKSPACE:str = r'D:/A日常/大学/毕业设计/code/yui/'
  # MAX_INPUTS_LENGTH:int = 520
  # 此时对应一帧长度为 530*128/16000 = 4.24s

  # train
  TRAIN_STEPS:int = 200
  DEVICE:str = 'cpu'


@dataclasses.dataclass(frozen=True)
class ProConfig(BaseConfig):
  # io
  DATASET_DIR:str = r'/content/maestro-v3.0.0/'
  WORKSPACE:str = r'/content/'

  # train
  TRAIN_STEPS:int = 400000
  DEVICE:str = 'cuda'


cf = DevConfig()


if __name__ == '__main__':
  cf = DevConfig()
  print(cf)
