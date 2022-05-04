import collections
import functools
import logging
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, TypeVar
import re

import numpy as np
import note_seq
import mir_eval
import pretty_midi
import sklearn

import event_codec
import note_sequences
from config.data import YuiConfig
import vocabularies


S = TypeVar('S')
T = TypeVar('T')

# 将模型输出进行detokenize并且去掉eos_id及其后面的内容
def detokenize(
  predictions: Sequence[int],
  config: YuiConfig,
  vocab:vocabularies.Vocabulary
) -> np.ndarray:

  tokens = vocab.decode(predictions)
  # decode 将 EOS_ID -> DECODED_EOS_ID
  tokens = np.asarray(tokens, dtype=np.int32)
  if config.DECODED_EOS_ID in tokens:
    tokens = tokens[:np.argmax(tokens == config.DECODED_EOS_ID)]
    # 找decoded_eos_id第一次出现的地方，只保留这之前的部分；即 True > False
    # 实际上 vocab.decode 已经去掉了eos后面的部分，这里这是去掉了EOS

  return tokens


def decode_events(
    state: event_codec.DS,
    tokens: np.ndarray,
    start_time: float,
    max_time: Optional[float],
    codec: event_codec.Codec,
    decode_event_fn: Callable[[event_codec.DS, float, event_codec.Event, event_codec.Codec], None],
) -> Tuple[int, int]:
  """Decode a series of tokens, maintaining a decoding state object.

  Args:
    state: Decoding state object; will be modified in-place.
    tokens: event tokens to convert, should be processed by preprocessors.detokenize.
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
      # 在此之前应先经过 preprocessors.detokenize 处理
      event = codec.decode_event_index(token)
    except ValueError:
      invalid_events += 1
      continue
  
    if event.type == 'shift':
      cur_steps += event.value
      # 考虑到shift事件可能连续出现(RLE编码后)，将其都加上
      cur_time = start_time + cur_steps / codec.steps_per_second
      # 遇到shift事件就将其累加计算当前时间
      if max_time and cur_time > max_time:
        dropped_events = len(tokens) - token_idx
        break
      # 超出下一段的开始时间，丢弃超出部分的tokens

    else:
      cur_steps = 0
      # shift事件代表1step，extract_features截取的每一段target再经过RLE后shift在段内自己累计，
      # 但由于加上了start_time，因此这里是相对于整首曲子的时间（绝对时间）
      # 故遇到非shift事件就将其清零，下次遇到shift再设置
      try:
        decode_event_fn(state, cur_time, event, codec)
        # 遇到非shift事件，将其更新到state里面
      except ValueError:
        invalid_events += 1
        logging.debug(
          f'Got invalid event when decoding event {event} at time {cur_time}. \
            Invalid event counter now at {invalid_events}.', 
          exc_info=True
        )
        # exc_info=True: with traceback and message
        continue
      
  return invalid_events, dropped_events


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

  for i, pred in enumerate(sorted_predictions):
    begin_segment_fn(state) # 此处无用，没有重新赋值: lambda state: None

    # Depending on the audio token hop length, each symbolic token could be
    # associated with multiple audio frames. Since we split up the audio frames
    # into segments for prediction, this could lead to overlap. To prevent
    # overlap issues, ensure that the current segment does not make any
    # predictions for the time period covered by the subsequent segment.
    max_decode_time = None
    if i < len(sorted_predictions) - 1:
      max_decode_time = sorted_predictions[i + 1]['start_time']
      # 以下一段的开始时间作为本段解码的截止时间，本段超出这个时间的token会被丢弃
      # 或许就是这样解决片段重叠的问题！

    logging.debug(f'in decode_and_combine_predictions<{i}>, {pred["start_time"]=}, {max_decode_time=}')

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
  encoding_spec: event_codec.EventEncodingSpec
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
      decode_events,
      codec=codec,
      decode_event_fn=encoding_spec.decode_event_fn
    ),
    flush_state_fn=encoding_spec.flush_decoding_state_fn
  )
  # 该函数生成的ns不含有拍号、速度等信息，TODO 之后添加

  # encoding_spec = event_codec.EventEncodingSpec(
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


def get_prettymidi_pianoroll(ns: note_seq.NoteSequence, fps: float=YuiConfig.PIANOROLL_FPS, is_drum: bool=False):
  """Convert NoteSequence to pianoroll through pretty_midi."""

  for note in ns.notes:
    if is_drum or note.end_time - note.start_time < 0.05:
      # Give all drum notes a fixed length, and all others a min length
      note.end_time = note.start_time + 0.05

  pm = note_seq.note_sequence_to_pretty_midi(ns)
  end_time = pm.get_end_time()
  cc = [
      # all sound off
      pretty_midi.ControlChange(number=120, value=0, time=end_time),
      # all notes off
      pretty_midi.ControlChange(number=123, value=0, time=end_time)
  ]
  pm.instruments[0].control_changes = cc
  if is_drum:
    # If inst.is_drum is set, pretty_midi will return an all zero pianoroll.
    for inst in pm.instruments:
      inst.is_drum = False
  pianoroll = pm.get_piano_roll(fs=fps)  # (128, times.shape[0])
  return pianoroll


def pianoroll_to_upr(pr: np.ndarray) -> list[str]:
  """将稀疏的pretty_midi钢琴卷帘处理成便于ui使用的格式
  仍然128行，但每行的每个音符条用 't-\d v-\d c-\d' 分别表示开始时刻、力度以及个数（持续时间）
  !此时读取顺序从 y=127 -> y=0，跟ui中卷帘排列顺序相反
  return: ['t0v22c2 t5v26c1 t6v51c2', 't4v24c6', ...]
  """

  # assert len(pr.shape)==2 and pr.shape[0]==128
  upr = []
  for r in pr:
    data = np.nonzero(r)[0]
    # [2, 24, 25, 26, 27, 51, 52, ..] 代表每行非零元素下标
    # [0] 因为按行处理返回的是单元素的元组
    if not np.any(data):
      upr.append('')
      continue

    last_i = -9
    last_vel = 0
    vel_cnt = 0
    line = []
    cur_tv = ''
    for i in data:
      cur_vel = int(r[i])
      # 力度必为整数
      if i-1 > last_i or cur_vel != last_vel:
        if last_vel != 0:
          line.append(f'{cur_tv}c{vel_cnt}')
        cur_tv = f't{i}v{cur_vel}'
        vel_cnt = 1
        # 间隔后产生新音符或音符以不同力度重新发出
      else:
        vel_cnt += 1
        # 该音正在持续
      last_vel = cur_vel
      last_i = i

    line.append(f'{cur_tv}c{vel_cnt}')
    # 最后一个音必须特别处理
    upr.append(' '.join(line))

  return upr


def upr_to_pianoroll(upr: list) -> np.ndarray:
  """将upr还原为 prettyMIDI.pianoroll
  upr: ['t0v22c2 t5v26c1 t6v51c2', 't4v24c6', ...]
  """

  tcv_pattern = re.compile(r't(\d+)v(\d+)c(\d+)')
  pianoroll = []
  max_line_len = 0
  for r in upr:
    if not r:
      pianoroll.append([])
      continue

    line = []
    for n in r.split(' '):
      m = tcv_pattern.match(n)
      t, v, c = tuple(map(lambda x: int(x), m.groups()))
      if (line_len := len(line)) >= t:
        line = line[:t]
      else:
        line.extend([0] * (t-line_len-1))
      # 当前音符开始前一个还没结束就截断，离得远就填0
      line.extend([v] * c)
      # 若力度相同的音符向后重叠会被当做一整个长音符

    max_line_len = max(max_line_len, len(line))
    pianoroll.append(line)
  
  for line in pianoroll:
    line.extend([0] * (max_line_len-len(line)))

  return np.asarray(pianoroll)


def piano_roll_to_pretty_midi(piano_roll: np.ndarray, fs: float=100, program: int=0) -> pretty_midi.PrettyMIDI:
    """Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    !both prettymidi to pianoroll and pianoroll to prettymidi are lossy

    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    """

    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)
    # 两个数组，分别代表时间跟音高维度力度变动事件所在下标; diff默认仅计算时间维度

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
      # use time + 1 because of padding above
      velocity = piano_roll[note, time + 1]
      time = time / fs
      if velocity > 0:
        if prev_velocities[note] == 0:
          note_on_time[note] = time
          prev_velocities[note] = velocity
      else:
        pm_note = pretty_midi.Note(
          velocity=prev_velocities[note],
          pitch=note,
          start=note_on_time[note],
          end=time
        )
        instrument.notes.append(pm_note)
        prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


def frame_metrics(
  ref_pianoroll: np.ndarray,
  est_pianoroll: np.ndarray,
  velocity_threshold: int
) -> Tuple[float, float, float]:
  """Frame Precision, Recall, and F1."""

  # Pad to same length
  if ref_pianoroll.shape[1] > est_pianoroll.shape[1]:
    diff = ref_pianoroll.shape[1] - est_pianoroll.shape[1]
    est_pianoroll = np.pad(est_pianoroll, [(0, 0), (0, diff)], mode='constant')
  elif est_pianoroll.shape[1] > ref_pianoroll.shape[1]:
    diff = est_pianoroll.shape[1] - ref_pianoroll.shape[1]
    ref_pianoroll = np.pad(ref_pianoroll, [(0, 0), (0, diff)], mode='constant')

  # For ref, remove any notes that are too quiet
  ref_frames_bool = ref_pianoroll > velocity_threshold
  # For est, keep all predicted notes.
  est_frames_bool = est_pianoroll > 0

  precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
    ref_frames_bool.flatten(),
    est_frames_bool.flatten(),
    labels=[True, False]
  )

  return precision[0], recall[0], f1[0]
  # 只取label=True的结果


def _note_onset_tolerance_sweep(
  est: tuple[np.ndarray, np.ndarray],
  ref: tuple[np.ndarray, np.ndarray],
  tolerances: Sequence[float] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
) -> Mapping[str, float]:
  """Compute note precision/recall/F1 across a range of tolerances."""

  est_intervals, est_pitches = est
  ref_intervals, ref_pitches = ref
  scores = {}

  for tol in tolerances:
    precision, recall, f_measure, _ = mir_eval.transcription.precision_recall_f1_overlap(
      ref_intervals=ref_intervals, ref_pitches=ref_pitches,
      est_intervals=est_intervals, est_pitches=est_pitches,
      offset_ratio=None,
      onset_tolerance=tol, offset_min_tolerance=tol,
    )
    
    scores[f'Onset & offset precision ({tol})'] = precision
    scores[f'Onset & offset recall ({tol})'] = recall
    scores[f'Onset & offset F1 ({tol})'] = f_measure

  return scores


def event_tokens_to_ns(
  events: list[tuple[float, Sequence]],
  codec: event_codec.Codec,
  encoding_spec: event_codec.EventEncodingSpec = note_sequences.NoteEncodingSpec
) -> Mapping[str, Any]:
  """将事件tokens，即 proprocessors.encode_events 输出转换为 note_sequences"""
  # 取代 predictions_to_ns 和 decode_and_combine_predictions，但参数不同，且保留二者
  
  init_state_fn = encoding_spec.init_decoding_state_fn
  begin_segment_fn = encoding_spec.begin_decoding_segment_fn
  decode_tokens_fn = functools.partial(
    decode_events,
    codec=codec,
    decode_event_fn=encoding_spec.decode_event_fn
  )
  flush_state_fn = encoding_spec.flush_decoding_state_fn

  sorted_events = sorted(events, key=lambda x: x[0])  # 以start_Time为基准排序
  state = init_state_fn()
  # 一个dataclass: NoteDecodingState，内含 ns, current_time, current_velocity, current_program, active_pitches
  # 其中ns不在下方每个循环中不断更新、添加，整合这首曲子的所有tokens
  total_invalid_events = 0
  total_dropped_events = 0

  for i, (start_time, tokens) in enumerate(sorted_events):
    begin_segment_fn(state) # 此处无用，没有重新赋值: lambda state: None
    max_decode_time = None
    if i < len(sorted_events) - 1:
      max_decode_time = sorted_events[i + 1][0]
      # 以下一段的开始时间作为本段解码的截止时间，本段超出这个时间的token会被丢弃
      # 或许就是这样解决片段重叠的问题！

    logging.debug(f'in decode_and_combine_predictions<{i}>, {start_time=}, {max_decode_time=}')

    invalid_events, dropped_events = decode_tokens_fn(state, tokens, start_time, max_decode_time)
    total_invalid_events += invalid_events
    total_dropped_events += dropped_events

  # logging.debug(f'after decode_and_combine_predictions, {state=}')
  return {
    'ns': flush_state_fn(state),
    'invalid_events': total_invalid_events,
    'dropped_events': total_dropped_events,
    # 记录无效与丢弃的事件数量
  }


def calc_metrics(
  pred_map: dict[int, list],
  target_map: dict[int, list],
  codec: event_codec.Codec,
  frame_fps: float = YuiConfig.PIANOROLL_FPS,
  frame_velocity_threshold: int = 30,
  use_offsets = True,
  use_velocities = True,
) -> dict[str, Any]:
  """Compute mir_eval transcription metrics.
  pred_map: {audio_id: [(start_time, tokens), ...]}
  target_map: {audio_id: [(start_time, tokens), ...]}
  frame_fps: 用于pretty_midi.get_piano_roll的分辨率，越大pianoroll越精细，而 1/62.5==0.016s -> 两帧(256/16kHz)一次采样
  """
  # logging.info(pred_map)
  # logging.info(target_map)
  # return

  # 产生pred和target的ns
  pred_target_pairs = []
  idx_list = []
  for k in pred_map:
    assert k in target_map
    pred_dict = event_tokens_to_ns(pred_map[k], codec)
    target_dict = event_tokens_to_ns(target_map[k], codec)
    pred_target_pairs.append((pred_dict, target_dict))
    idx_list.append(k)
  # 丢弃audio_id，反正所有曲子都要处理

  scores = collections.defaultdict(list)
  pianorolls = []
  est_ns_list = []
  for pred_dict, target_dict in pred_target_pairs:
    scores['Invalid events'].append(pred_dict['invalid_events'])
    scores['Dropped events'].append(pred_dict['dropped_events'])
    est_ns = pred_dict['ns']
    ref_ns = target_dict['ns']
    est_ns_list.append(est_ns)

    est_intervals, est_pitches, est_velocities = note_seq.sequences_lib.sequence_to_valued_intervals(est_ns)
    ref_intervals, ref_pitches, ref_velocities = note_seq.sequences_lib.sequence_to_valued_intervals(ref_ns)

    # Precision / recall / F1 using onsets (and pitches) only.
    precision, recall, f_measure, avg_overlap_ratio = (
      mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals=ref_intervals,
        ref_pitches=ref_pitches,
        est_intervals=est_intervals,
        est_pitches=est_pitches,
        offset_ratio=None
      )
    )
    # del avg_overlap_ratio
    scores['Onset precision'].append(precision)
    scores['Onset recall'].append(recall)
    scores['Onset F1'].append(f_measure)
    scores['Onset AOR'].append(avg_overlap_ratio)

    if use_offsets:
      # Precision / recall / F1 using onsets and offsets.
      precision, recall, f_measure, avg_overlap_ratio = (
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=ref_pitches,
            est_intervals=est_intervals,
            est_pitches=est_pitches
        )
      )
      # del avg_overlap_ratio
      scores['Onset & offset precision'].append(precision)
      scores['Onset & offset recall'].append(recall)
      scores['Onset & offset F1'].append(f_measure)
      scores['Onset & offset AOR'].append(avg_overlap_ratio)

    if use_velocities:
      # Precision / recall / F1 using onsets and velocities (no offsets).
      precision, recall, f_measure, avg_overlap_ratio = (
        mir_eval.transcription_velocity.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=ref_pitches,
            ref_velocities=ref_velocities,
            est_intervals=est_intervals,
            est_pitches=est_pitches,
            est_velocities=est_velocities,
            offset_ratio=None
        )
      )
      scores['Onset & velocity precision'].append(precision)
      scores['Onset & velocity recall'].append(recall)
      scores['Onset & velocity F1'].append(f_measure)
      scores['Onset & velocity AOR'].append(avg_overlap_ratio)

    if use_offsets and use_velocities:
      # Precision / recall / F1 using onsets, offsets, and velocities.
      precision, recall, f_measure, avg_overlap_ratio = (
        mir_eval.transcription_velocity.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=ref_pitches,
            ref_velocities=ref_velocities,
            est_intervals=est_intervals,
            est_pitches=est_pitches,
            est_velocities=est_velocities
        )
      )
      scores['Onset & offset & velocity precision'].append(precision)
      scores['Onset & offset & velocity recall'].append(recall)
      scores['Onset & offset & velocity F1'].append(f_measure)
      scores['Onset & offset & velocity AOR'].append(avg_overlap_ratio)
    # 根据不同对象、组合计算多组指标

    # Calculate framewise metrics.
    is_drum = all([n.is_drum for n in ref_ns.notes])
    ref_pr = get_prettymidi_pianoroll(ref_ns, frame_fps, is_drum=is_drum)
    est_pr = get_prettymidi_pianoroll(est_ns, frame_fps, is_drum=is_drum)
    pianorolls.append((est_pr, ref_pr))
    frame_precision, frame_recall, frame_f1 = frame_metrics(ref_pr, est_pr, velocity_threshold=frame_velocity_threshold)
    scores['Frame Precision'].append(frame_precision)
    scores['Frame Recall'].append(frame_recall)
    scores['Frame F1'].append(frame_f1)

    # 针对 onset/offset 的考虑各种 tolerances 的指标
    for name, score in _note_onset_tolerance_sweep(
      est=(est_intervals, est_pitches), 
      ref=(ref_intervals, ref_pitches)
    ).items():
      scores[name].append(score)

  mean_scores = {k: np.mean(v) for k, v in scores.items()}
  score_histograms = {f'{k} [hist]': np.asarray(v) for k, v in scores.items()}
  extra_map = {
    'idx_list': np.asarray(idx_list),
    'pianorolls': pianorolls,
    'pred_ns_list': est_ns_list,
  }
  return mean_scores | score_histograms | extra_map
