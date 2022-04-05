import functools
import logging
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy as np

import event_codec
import note_sequences
import run_length_encoding
from config.data import YuiConfig
import vocabularies


S = TypeVar('S')
T = TypeVar('T')


# 将模型输出进行detokenize并且去掉eos_id及其后面的内容
def detokenize(
  predictions: Sequence[int],
  config: YuiConfig,
  vocab:vocabularies.GenericTokenVocabulary
) -> np.ndarray:

  tokens = vocab.decode(predictions)
  # decode 将 EOS_ID -> DECODED_EOS_ID
  tokens = np.asarray(tokens, dtype=np.int32)
  if config.DECODED_EOS_ID in tokens:
    tokens = tokens[:np.argmax(tokens == config.DECODED_EOS_ID)]

  return tokens


def decode_and_combine_predictions(
  predictions: Sequence[Mapping[str, Any]],
  init_state_fn: Callable[[], S],
  begin_segment_fn: Callable[[S], None],
  decode_tokens_fn: Callable[
    [S, Sequence[int], int, Optional[int]],
    Tuple[int, int]
  ],
  flush_state_fn: Callable[[S], T]
) -> Tuple[T, int, int]:
  """Decode and combine a sequence of predictions to a full result.

  For time-based events, this usually means concatenation.

  Args:
    predictions: List of predictions, each of which is a dictionary containing
        estimated tokens ('est_tokens') and start time ('start_time') fields.
    init_state_fn: Function that takes no arguments and returns an initial
        decoding state.
    begin_segment_fn: Function that updates the decoding state at the beginning
        of a segment.
    decode_tokens_fn: Function that takes a decoding state, estimated tokens
        (for a single segment), start time, and max time, and processes the
        tokens, updating the decoding state in place. Also returns the number of
        invalid and dropped events for the segment.
    flush_state_fn: Function that flushes the final decoding state into the
        result.

  Returns:
    result: The full combined decoding.
    total_invalid_events: Total number of invalid event tokens across all
        predictions.
    total_dropped_events: Total number of dropped event tokens across all
        predictions.
  """

  sorted_predictions = sorted(predictions, key=lambda pred: pred['start_time'])
  state = init_state_fn()
  # 一个dataclass: NoteDecodingState，内含 ns, current_time, current_velocity, current_program, active_pitches
  # 其中ns不在下方每个循环中不断更新、添加，整合这首曲子的所有tokens
  total_invalid_events = 0
  total_dropped_events = 0

  for pred_idx, pred in enumerate(sorted_predictions):
    begin_segment_fn(state) # 此处无用，没有重新赋值: lambda state: None

    # Depending on the audio token hop length, each symbolic token could be
    # associated with multiple audio frames. Since we split up the audio frames
    # into segments for prediction, this could lead to overlap. To prevent
    # overlap issues, ensure that the current segment does not make any
    # predictions for the time period covered by the subsequent segment.
    max_decode_time = None
    if pred_idx < len(sorted_predictions) - 1:
      max_decode_time = sorted_predictions[pred_idx + 1]['start_time']
      # 以下一段的开始时间作为本段解码的截止时间，本段超出这个时间的token会被丢弃
      # 或许就是这样解决片段重叠的问题！

    logging.info(f'in decode_and_combine_predictions<{pred_idx}>, {pred["start_time"]=}, {max_decode_time=}')

    invalid_events, dropped_events = decode_tokens_fn(
      state, pred['est_tokens'], pred['start_time'], max_decode_time
    )

    total_invalid_events += invalid_events
    total_dropped_events += dropped_events

  # logging.info(f'after decode_and_combine_predictions, {state=}')
  return flush_state_fn(state), total_invalid_events, total_dropped_events


def predictions_to_ns(
  predictions: Sequence[Mapping[str, Any]],
  codec: event_codec.Codec,
  encoding_spec: note_sequences.NoteEncodingSpecType
) -> Mapping[str, Any]:
  """Convert a sequence of predictions to a combined NoteSequence.

  predictions: Sequence[Mapping[str, Any]] = {
    'est_tokens': tokens,
    'start_time': start_time,
    'raw_inputs': []
  }
  tokens：经过RLE、encode的event_codec.Event，1-d的整数序列
  """
  
  ns, total_invalid_events, total_dropped_events = decode_and_combine_predictions(
    predictions=predictions,
    init_state_fn=encoding_spec.init_decoding_state_fn,
    begin_segment_fn=encoding_spec.begin_decoding_segment_fn,
    decode_tokens_fn=functools.partial(
      run_length_encoding.decode_events,
      codec=codec,
      decode_event_fn=encoding_spec.decode_event_fn
    ),
    flush_state_fn=encoding_spec.flush_decoding_state_fn
  )
  # 该函数生成的ns不含有拍号、速度等信息，TODO 之后添加

  # encoding_spec = NoteEncodingSpecType(
  #   init_encoding_state_fn=lambda: None,
  #   encode_event_fn=note_event_data_to_events,
  #   encoding_state_to_events_fn=None,
  #   init_decoding_state_fn=NoteDecodingState,
  #   begin_decoding_segment_fn=lambda state: None,
  #   decode_event_fn=decode_note_event,
  #   flush_decoding_state_fn=flush_note_decoding_state
  # )

  # Also concatenate raw inputs from all predictions.
  # sorted_predictions = sorted(predictions, key=lambda pred: pred['start_time'])
  # raw_inputs = np.concatenate([pred['raw_inputs'] for pred in sorted_predictions], axis=0)
  # start_times = [pred['start_time'] for pred in sorted_predictions]

  return {
    # 'raw_inputs': raw_inputs,
    # 'start_times': start_times,
    'est_ns': ns,
    'est_invalid_events': total_invalid_events,
    'est_dropped_events': total_dropped_events,
    # 记录无效与丢弃的事件数量
  }
