import dataclasses
from typing import Sequence

from sklearn import preprocessing

import note_sequences


@dataclasses.dataclass(frozen=True)
class BaseConfig:
    RANDOM_SEED:int = 233
    NOTES_NUM:int = 128    # NUMBER OF NOTES OF PIANO
    VELOCITY_SCALE:int = 128
    TIME_SCALE:int = 6000

    # preprocess
    NUM_VELOCITY_BINS = 127
    TASK_INPUT_LENGTHS:int =  512
    TASK_TARGET_LENGTHS:int = 1024
    MAX_TOKENS_PER_SEGMENT:int = 2000
    PROGRAM_GRANULARITY = 'flat'

    # spectrogram
    SAMPLE_RATE:int = 16000
    HOP_WIDTH:int = 128
    NUM_MEL_BINS:int = 512
    FFT_SIZE:int = 2048  # fft_window_size and hann_window_size
    MEL_LO_HZ:float = 20.0
    MEL_HI_HZ:float = 7600.0

    # train
    TRAIN_STEPS = 400000

    @property
    def frames_per_second(self):
        return self.SAMPLE_RATE / self.HOP_WIDTH


@dataclasses.dataclass(frozen=True)
class DevConfig(BaseConfig):
    DATASET_DIR:str = r'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0/'
    WORKSPACE:str = r'D:/A日常/大学/毕业设计/dataset/'


@dataclasses.dataclass(frozen=True)
class ProConfig(BaseConfig):
    DATASET_DIR:str = r'./maestro-v3.0.0/'
    WORKSPACE:str = r'./'


if __name__ == '__main__':
    cf = DevConfig()
    print(cf)
