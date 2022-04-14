import dataclasses
import math
from typing import Callable, Optional, Sequence

import event_codec
from config.data import YuiConfig

import note_seq
import numpy as np


def num_velocity_bins_from_codec(codec: event_codec.Codec):
  """Get number of velocity bins from event codec."""
  lo, hi = codec.event_type_range('velocity')
  return hi - lo


def velocity_to_bin(velocity, num_velocity_bins):
  """将velocity乘一个比例系数，缩放到另一个区间"""
  if velocity == 0:
    return 0
  else:
    return math.ceil(num_velocity_bins * velocity / note_seq.MAX_MIDI_VELOCITY)


def bin_to_velocity(velocity_bin, num_velocity_bins):
  if velocity_bin == 0:
    return 0
  else:
    return int(note_seq.MAX_MIDI_VELOCITY * velocity_bin / num_velocity_bins)


def drop_programs(tokens, codec: event_codec.Codec):
  """Drops program change events from a token sequence."""
  # https://www.recordingblogs.com/wiki/midi-program-change-message
  # tells a MIDI device that at a certain time certain program should be selected for one of the MIDI channels. Usually this means that an instrument will be selected for the MIDI channel and notes that follow this message will be played with the selected instrument.
  # program change似乎代表乐器的变换，而0x00指的是Acoustic grand piano
  # 因此对于MAESTRO数据集要去掉program change event，所有program都改为0

  min_program_id, max_program_id = codec.event_type_range('program')
  return tokens[(tokens < min_program_id) | (tokens > max_program_id)]


def programs_to_midi_classes(tokens, codec):
  """Modifies program events to be the first program in the MIDI class."""
  min_program_id, max_program_id = codec.event_type_range('program')
  is_program = (tokens >= min_program_id) & (tokens <= max_program_id)
  return np.where(
      is_program,
      min_program_id + 8 * ((tokens - min_program_id) // 8),
      tokens
  )


@dataclasses.dataclass
class ProgramGranularity:
  # both tokens_map_fn and program_map_fn should be idempotent
  tokens_map_fn: Callable[[Sequence[int], event_codec.Codec], Sequence[int]]
  program_map_fn: Callable[[int], int]


PROGRAM_GRANULARITIES = {
    # "flat" granularity; drop program change tokens and set NoteSequence
    # programs to zero
    'flat': ProgramGranularity(
        tokens_map_fn=drop_programs,
        program_map_fn=lambda program: 0),

    # map each program to the first program in its MIDI class
    'midi_class': ProgramGranularity(
        tokens_map_fn=programs_to_midi_classes,
        program_map_fn=lambda program: 8 * (program // 8)),

    # leave programs as is
    'full': ProgramGranularity(
        tokens_map_fn=lambda tokens, codec: tokens,
        program_map_fn=lambda program: program)
}


def build_codec(config: YuiConfig):
  """Build event codec."""
  event_ranges = [
    event_codec.EventRange('pitch', note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_PITCH),
    # velocity bin 0 is used for note-off
    event_codec.EventRange('velocity', 0, config.NUM_VELOCITY_BINS),
    # used to indicate that a pitch is present at the beginning of a segment
    # (only has an "off" event as when using ties all pitch events until the
    # "tie" event belong to the tie section)
    # event_codec.EventRange('tie', 0, 0),
    # event_codec.EventRange('program', note_seq.MIN_MIDI_PROGRAM, note_seq.MAX_MIDI_PROGRAM),
    # event_codec.EventRange('drum', note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_PITCH),
  ]

  return event_codec.Codec(
    max_shift_steps=config.max_shift_steps,
    steps_per_second=config.STEPS_PER_SECOND,
    event_ranges=event_ranges
  )


class Vocabulary:
  """Vocabulary with pass-through encoding of tokens."""

  def __init__(self, config: YuiConfig, regular_ids: int, extra_ids: int = 0):
    # The special tokens: 0=PAD, 1=EOS, and 2=UNK
    # extra_ids: The number of extra IDs to reserve.
    self._config = config
    self._num_special_tokens = 3
    self._num_regular_tokens = regular_ids
    self._extra_ids = extra_ids or 0

  @property
  def pad_id(self) -> int:
    return self._config.PAD_ID
  
  @property
  def eos_id(self) -> Optional[int]:
    return self._config.ENCODED_EOS_ID

  @property
  def unk_id(self) -> Optional[int]:
    return self._config.ENCODED_UNK_ID

  @property
  def extra_ids(self) -> int:
    return self._extra_ids

  @property
  def _base_vocab_size(self) -> int:
    """Vocabulary size, excluding extra ids but including PAD/EOS/UNK.

    Returns:
      an integer, the vocabulary size
    """
    return self._num_special_tokens + self._num_regular_tokens

  @property
  def vocab_size(self) -> int:
    """Vocabulary size, including extra ids."""
    return self._base_vocab_size + self.extra_ids

  def encode(self, token_ids: Sequence[int]):
    """Encode a list of tokens ids as a list of integers.

    To keep the first few ids for special tokens, increase ids by the number
    of special tokens.

    Args:
      token_ids: array of token ids.

    Returns:
      a list of integers (not terminated by EOS)
    """
    encoded = []
    for token_id in token_ids:
      if not 0 <= token_id < self._num_regular_tokens:
        raise ValueError(
          f'token_id {token_id} does not fall within valid range of '
          f'[0, {self._num_regular_tokens})'
        )
      encoded.append(token_id + self._num_special_tokens)
      # 将原有的数字token序列每个加上一个位移（特殊token的数量）得到新的数字token序列

    return encoded

  def decode(self, ids: Sequence[int]):
    """Decode a list of integers to a list of token ids.

    The special tokens of PAD and UNK as well as extra_ids will be
    replaced with DECODED_INVALID_ID in the output. If EOS is present, it will
    be the final token in the decoded output and will be represented by
    DECODED_EOS_ID.

    Args:
      ids: a list of integers

    Returns:
      a list of token ids.
    """
    # convert all the extra ids  to INVALID_ID
    clean_ids = list(ids)
    if self.unk_id is not None:
      clean_ids = [
        self.unk_id if i >= self._base_vocab_size else i
        for i in clean_ids
      ]
    if self.eos_id is not None and self.eos_id in clean_ids:
      clean_ids = clean_ids[:clean_ids.index(self.eos_id) + 1]

    def _decode_id(encoded_id):
      if encoded_id == self.eos_id:
        return self._config.DECODED_EOS_ID
      elif encoded_id < self._num_special_tokens:
        return self._config.DECODED_INVALID_ID
      elif encoded_id >= self._base_vocab_size:
        return self._config.DECODED_INVALID_ID
        # 除eos以外（unk, pad）以及超出词表的都判定为无效
      else:
        return encoded_id - self._num_special_tokens

    ids = [_decode_id(i) for i in clean_ids]
    return ids


# def vocabulary_from_codec(codec: event_codec.Codec) -> Vocabulary:
#   # return GenericVocabulary(codec.num_classes, extra_ids=cf.EXTRA_IDS)
#   return Vocabulary(codec.num_classes, extra_ids=100)
#   # DEFAULT_EXTRA_IDS = 100


def num_embeddings(vocabulary: Vocabulary) -> int:
  """Vocabulary size as a multiple of 128 for TPU efficiency."""
  return 128 * math.ceil(vocabulary.vocab_size / 128)
