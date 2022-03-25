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
  PROGRAM_GRANULARITY:str = 'flat'

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
  DECODED_EOS_ID:int = -1
  DECODED_INVALID_ID:int = -2
  STEPS_PER_SECOND:int = 100  # 每秒的处理步数，相当于对音符处理的精度
  MAX_SHIFT_SECONDS:int = 10
  NUM_VELOCITY_BINS:int = 127

  @property
  def frames_per_second(self):
    return self.SAMPLE_RATE // self.FRAME_SIZE

  @property
  def segment_second(self):
    return (self.MAX_INPUTS_LENGTH * self.FRAME_SIZE) / self.SAMPLE_RATE
  # 以MAX_INPUTS_LENGTH作为一段长度，若小于该值则之后进行pad
  
  @property
  def max_shift_steps(self):
    return min(int(self.segment_second)+1, self.MAX_SHIFT_SECONDS) * self.STEPS_PER_SECOND


@dataclasses.dataclass(frozen=True)
class DevConfig(BaseConfig):
  # io
  DATASET_DIR:str = r'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0/'
  WORKSPACE:str = r'D:/A日常/大学/毕业设计/code/yui/'
  MAX_INPUTS_LENGTH:int = 512
  # 此时对应一帧长度为 530*128/16000 = 4.24s

  # vocabulary
  STEPS_PER_SECOND:int = 990  
  # 总之不能取1000，musesocre效果很差，原理不明，感觉本质都是一样，确实越大越精细，但是midi转五线谱方案毕竟很多...或许跟转换用的软件也有关
  # MAX_SHIFT_SECONDS:int = 6  # 取小于 segment_second 的数，毕竟是每段内相对的

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
