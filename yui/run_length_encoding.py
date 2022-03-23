import dataclasses
from typing import Any, Callable, Tuple, Optional, Sequence, TypeVar
from absl import logging

import numpy as np

import event_codec


Event = event_codec.Event

# These should be type variables, but unfortunately those are incompatible with
# dataclasses.
EventData = Any
EncodingState = Any
DecodingState = Any
DecodeResult = Any

T = TypeVar('T', bound=EventData)
ES = TypeVar('ES', bound=EncodingState)
DS = TypeVar('DS', bound=DecodingState)


@dataclasses.dataclass
class EventEncodingSpec:
  """Spec for encoding events."""
  # initialize encoding state
  init_encoding_state_fn: Callable[[], EncodingState]
  # convert EventData into zero or more events, updating encoding state
  encode_event_fn: Callable[
    [EncodingState, EventData, event_codec.Codec],
    Sequence[event_codec.Event]
  ]
  # convert encoding state (at beginning of segment) into events
  encoding_state_to_events_fn: Optional[
    Callable[[EncodingState], Sequence[event_codec.Event]]
  ]
  # create empty decoding state
  init_decoding_state_fn: Callable[[], DecodingState]
  # update decoding state when entering new segment
  begin_decoding_segment_fn: Callable[[DecodingState], None]
  # consume time and Event and update decoding state
  decode_event_fn: Callable[
    [DecodingState, float, event_codec.Event, event_codec.Codec], 
    None
  ]
  # flush decoding state into result
  flush_decoding_state_fn: Callable[[DecodingState], DecodeResult]


def encode_and_index_events(
    state: ES,
    event_times: Sequence[float],
    event_values: Sequence[T],
    encode_event_fn: Callable[[ES, T, event_codec.Codec], Sequence[event_codec.Event]],
    codec: event_codec.Codec,
    frame_times: Sequence[float],
    encoding_state_to_events_fn: Optional[Callable[[ES], Sequence[event_codec.Event]]] = None,
) -> Tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int], Sequence[int]]:
  """Encode a sequence of timed events and index to audio frame times.
  将离散的、时间无关的、直接从note_sequences中提取的NoteEventData转换成
  时间连续的(按时间顺序排列事件，不存在音符的时间点用shift占位)、音高,力度,音色等分离的event_codec.Event

  Encodes time shifts as repeated single step shifts for later run length
  encoding.

  Optionally, also encodes a sequence of "state events", keeping track of the
  current encoding state at each audio frame. This can be used e.g. to prepend
  events representing the current state to a targets segment.

  Args:
    state: Initial event encoding state.
    event_times: Sequence of event times.
    event_values: Sequence of event values.
    encode_event_fn: Function that transforms event value into a sequence of one
        or more event_codec.Event objects.
    codec: An event_codec.Codec object that maps Event objects to indices.
    frame_times: Time for every audio frame.
    encoding_state_to_events_fn: Function that transforms encoding state into a
        sequence of one or more event_codec.Event objects.

  Returns:
    events: Encoded events and shifts.
    event_start_indices: Corresponding start event index for every audio frame.
        Note: one event can correspond to multiple audio indices due to sampling
        rate differences. This makes splitting sequences tricky because the same
        event can appear at the end of one sequence and the beginning of
        another.
    event_end_indices: Corresponding end event index for every audio frame. Used
        to ensure when slicing that one chunk ends where the next begins. Should
        always be true that event_end_indices[i] = event_start_indices[i + 1].
    state_events: Encoded "state" events representing the encoding state before
        each event.
    state_event_indices: Corresponding state event index for every audio frame.
  """

  indices = np.argsort(event_times, kind='stable')
  event_steps = [round(event_times[i] * codec.steps_per_second) for i in indices]
  # 如 [87, 87, 98, 99, 106, 108...., 4516]]，midi中记录的音符开始/结束时间，只是数值被放大了100倍，原本应是0.87s，总时长45.16s...
  event_values = [event_values[i] for i in indices]
  # 按照时间重新对事件排序，同时时间排序并转换为step
  # 根据note_sequences.note_sequence_to_onsets_and_offsets_and_programs，这里steps、values都包含了音符开始和结束

  events = []
  state_events = []
  event_start_indices = []
  state_event_indices = []

  cur_step = 0
  cur_event_idx = 0
  cur_state_event_idx = 0

  def fill_event_start_indices_to_cur_step():
    # 根据cur_event_idx填写event_start_indices
    # TODO 待优化
    while(len(event_start_indices) < len(frame_times) and frame_times[len(event_start_indices)] < cur_step / codec.steps_per_second):
      event_start_indices.append(cur_event_idx)
      state_event_indices.append(cur_state_event_idx)

  for event_step, event_value in zip(event_steps, event_values):
    # 当前cur_step距离下一个事件event_step还有一定距离，就用shift标记
    while event_step > cur_step:
      events.append(codec.encode_event(Event(type='shift', value=1)))
      cur_step += 1
      fill_event_start_indices_to_cur_step()
      cur_event_idx = len(events)  # != "+=1"，下面events仍会修改
      cur_state_event_idx = len(state_events)

    if encoding_state_to_events_fn:
      # Dump state to state events *before* processing the next event, because
      # we want to capture the state prior to the occurrence of the event.
      for e in encoding_state_to_events_fn(state):
        state_events.append(codec.encode_event(e))

    # 将一个NoteEventData编码为一个或多个（在这里是三个）codec事件: program, velocity, pitch 
    # NoteEventData(pitch=24, velocity=93, program=0, is_drum=False, instrument=None)
    for e in encode_event_fn(state, event_value, codec):
      events.append(codec.encode_event(e))

  # After the last event, continue filling out the event_start_indices array.
  # The inequality is not strict because if our current step lines up exactly
  # with (the start of) an audio frame, we need to add an additional shift event
  # to "cover" that frame.
  # 后续不存在音符，继续用shift填满events
  while cur_step / codec.steps_per_second <= frame_times[-1]:
    events.append(codec.encode_event(Event(type='shift', value=1)))
    cur_step += 1
    fill_event_start_indices_to_cur_step()
    cur_event_idx = len(events)

  # Now fill in event_end_indices. We need this extra array to make sure that
  # when we slice events, each slice ends exactly where the subsequent slice
  # begins.
  # 直接用start_indices生成end_indices，保证二者首尾相接
  event_end_indices = event_start_indices[1:] + [len(events)]

  events = np.array(events, dtype=np.int32)
  state_events = np.array(state_events, dtype=np.int32)
  event_start_indices = np.array(event_start_indices, dtype=np.int32)
  event_end_indices = np.array(event_end_indices, dtype=np.int32)
  state_event_indices = np.array(state_event_indices, dtype=np.int32)

  return events, event_start_indices, event_end_indices, state_events, state_event_indices
  # events: [...1, 1258, 1225, 1049, 1258, 1220, 1061, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1258, 1228, 1037, 1, 1258, 1222, 1025...]
  # 没有音符的地方就1填充，有音符的地方必定是3的整数倍非1数字，连续3个分别代表 program, velocity, pitch 


def decode_events(
    state: DS,
    tokens: np.ndarray,
    start_time: int,
    max_time: Optional[int],
    codec: event_codec.Codec,
    decode_event_fn: Callable[[DS, float, event_codec.Event, event_codec.Codec], None],
) -> Tuple[int, int]:
  """Decode a series of tokens, maintaining a decoding state object.

  Args:
    state: Decoding state object; will be modified in-place.
    tokens: event tokens to convert.
    start_time: offset start time if decoding in the middle of a sequence.
    max_time: Events at or beyond this time will be dropped.
    codec: An event_codec.Codec object that maps indices to Event objects.
    decode_event_fn: Function that consumes an Event (and the current time) and
        updates the decoding state.

  Returns:
    invalid_events: number of events that could not be decoded.
    dropped_events: number of events dropped due to max_time restriction.
  """
  
  invalid_events = 0
  dropped_events = 0
  cur_steps = 0
  cur_time = start_time
  token_idx = 0

  for token_idx, token in enumerate(tokens):
    try:
      event = codec.decode_event_index(token)
    except ValueError:
      invalid_events += 1
      continue
  
    if event.type == 'shift':
      cur_steps += event.value
      # 考虑到shift事件可能连续出现，将其都加上
      cur_time = start_time + cur_steps / codec.steps_per_second
      # 遇到shift事件就将其累加计算当前时间
      if max_time and cur_time > max_time:
        dropped_events = len(tokens) - token_idx
        break
      # 超出下一段的开始时间，丢弃超出部分的tokens

    else:
      cur_steps = 0
      # shift事件并非单段tokens内独自计数(相对时间)，而是整首曲子范围内(绝对时间)
      # 因此当遇到非shift事件就将其清零，下次遇到shift再设置
      try:
        decode_event_fn(state, cur_time, event, codec)
        # 遇到非shift事件，将其更新到state里面
      except ValueError:
        invalid_events += 1
        logging.info(
          f'Got invalid event when decoding event {event} at time {cur_time}. \
            Invalid event counter now at {invalid_events}.', 
          exc_info=True
        )
        # exc_info=True: with traceback and message
        continue
      
  return invalid_events, dropped_events
