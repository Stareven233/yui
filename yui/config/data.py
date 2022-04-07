import dataclasses

import pretty_midi
pretty_midi.pretty_midi.MAX_TICK = 1e7
# 修改pretty_midi的最大tick数，便于读取某些数据集
# https://github.com/craffel/pretty-midi/issues/112


@dataclasses.dataclass(frozen=True)
class YuiConfig:
  RANDOM_SEED:int = 233

  # io
  DATASET_DIR:str = r'/content/maestro-v3.0.0/'
  DATAMETA_NAME:str = r'maestro-v3.0.0.csv'
  WORKSPACE:str = r'/content/'
  MAX_INPUTS_LENGTH:int =  512  # 指input的第一维，第二维是frame_size: (512, 128)
  MAX_TARGETS_LENGTH:int = 1024  # target第1维: (1024, )
  # MAX_INPUTS_LENGTH=512时实际切片为 512x128，约4.096s，假设最小音符间隔为10ms且同一时间就一个音符
  # 那也得 4.096*100*3(shift, velocity, pitch)，大概1200个事件，总之用512不够
  # MAX_SEGMENT_LENGTH:int =  2000
  PROGRAM_GRANULARITY:str = 'flat'

  # spectrogram
  SAMPLE_RATE:int = 16000
  FRAME_SIZE:int = 128
  HOP_WIDTH:int = FRAME_SIZE
  NUM_MEL_BINS:int = 512  # 作为嵌入维度应与模型d_model保持一致
  FFT_SIZE:int = 2048  # fft_window_size and hann_window_size
  MEL_LO_HZ:float = 20.0
  MEL_HI_HZ:float = SAMPLE_RATE / 2
  # 对应 librosa.filters.mel 的fmin跟fmax

  # vocabulary
  PAD_ID:int = 0
  ENCODED_EOS_ID:int = 1
  ENCODED_UNK_ID:int = 2
  EXTRA_IDS:int = 100  # 指额外id的数量
  DECODED_EOS_ID:int = -1
  DECODED_INVALID_ID:int = -2
  STEPS_PER_SECOND:int = 1000  # 每秒的处理步数，相当于对音符处理的精度
  # MAX_SHIFT_SECONDS:int = 6  # 取小于 segment_second 的数，毕竟是每段内相对的
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
    return int(min(self.segment_second, self.MAX_SHIFT_SECONDS) * self.STEPS_PER_SECOND) + 1
    # 取值<32767时可使用int16
  
  # train
  CUDA:bool = True
  BATCH_SIZE:int = 128  # 一个核16个
  NUM_WORKERS:int = 8
  NUM_EPOCHS:int = 20
  TRAIN_ITERATION:int = 500
  # 一共约140k个样本(以4.096s一个)，当batch_size=8，17k个iteration处理一遍数据，400k能将数据处理24遍
  LEARNING_RATE:float = 1e-3
  OVERFIT_PATIENCE:int = 5
  # train: 572,752s -> 139,833个样本，一个batch128大概 1100 iteration
  # validation: 69,946s -> 17,076个样本，134 iteration


@dataclasses.dataclass(frozen=True)
class YuiConfigPro(YuiConfig):
  ...


@dataclasses.dataclass(frozen=True)
class YuiConfigDev(YuiConfig):
  # io
  DATASET_DIR:str = r'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0/'
  DATAMETA_NAME:str = r'maestro-v3.0.0_tiny.csv'
  WORKSPACE:str = r'D:/A日常/大学/毕业设计/code/yui/'
  MAX_INPUTS_LENGTH:int =  16
  MAX_TARGETS_LENGTH:int = 64

  # spectrogram
  NUM_MEL_BINS:int = 12
  FFT_SIZE:int = 128

  # train
  CUDA:bool = False
  BATCH_SIZE:int = 8
  NUM_WORKERS:int = 0
  NUM_EPOCHS:int = 2
  TRAIN_ITERATION:int = 100
  # 当本地测试时batch=8，用tiny数据集，当batch_size=8，以0.128s一个样本
  # train: 414s -> 3,234个样本，404 iteration / epoch
  # validation: 175s -> 1,367个样本，170 iteration / epoch


if __name__ == '__main__':
  cf = YuiConfigDev()
  print(cf)
