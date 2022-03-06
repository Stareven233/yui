import dataclasses
from typing import Sequence

import note_sequences


@dataclasses.dataclass(frozen=True)
class BaseConfig:
    RANDOM_SEED:int = 233
    NOTES_NUM:int = 128    # NUMBER OF NOTES OF PIANO
    VELOCITY_SCALE:int = 128
    TIME_SCALE:int = 6000

    BEGIN_NOTE:int = 21     # MIDI NOTE OF A0, THE LOWEST NOTE OF A PIANO.
    SEGMENT_SECONDS:float = 10.	# TRAINING SEGMENT DURATION
    HOP_SECONDS:float = 1.
    FRAMES_PER_SECOND:int = 100

    DECODED_EOS_ID:int = -1
    DECODED_INVALID_ID:int = -2
    STEPS_PER_SECOND:int = 100
    MAX_SHIFT_SECONDS:int = 10
    NUM_VELOCITY_BINS:int = 127

    # spectrogram
    SAMPLE_RATE:int = 16000
    HOP_WIDTH:int = 128
    NUM_MEL_BINS:int = 512
    FFT_SIZE:int = 2048
    MEL_LO_HZ:float = 20.0

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
