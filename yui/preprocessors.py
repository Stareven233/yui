from calendar import c
import os
import time
from typing import Any, Mapping, Sequence
import logging

import numpy as np
import pandas as pd
import librosa
import note_seq
from torch import fmax, fmin
from transformers import MaxLengthCriteria

import event_codec, vocabularies
import note_sequences
import run_length_encoding
from utils import create_logging, get_feature_desc
from config.data import BaseConfig, DevConfig

# TODO 
# 0看featureconverter
# 1构思预处理流程，不能拘泥于细节
  # 考虑mp3和wav区别 -mp3压缩了高频
  # 利用h5py将几首歌预处理结果压在一起读取 -先不考虑
  # 每次sample记录读取到的文件id，便于中断后恢复训练
  # 定长不重叠切片，在dataset.getitem中进行预处理
  # 照着kaggle和bytedance进行预处理、做dataloader

# 2将midi进行预处理，再测试能否正确转换回来
# 3先按原来的featureconverter运行一遍，记录结果
# 4重写去掉tf依赖，对比原先的结果
# 4.1 将np变量统一成pytorch的
# 5数据集colab、kaggle都总会有办法，先下载下来，写好处理代码
# 6优化/减小模型，降低内存、运行时间
# 7利用github存储数据（或许可以结合h5）
# 8数据增强
  # 像bytedance那样改变音高、考虑踏板
  # 随机切分不等长且允许重叠的片段作为输入
  # 提取主旋律或统一音色后再训练


def upgrade_maestro(dataset_dir: str):
  """将maestro从v2.0.0升级到v3.0.0
  据v300更新说明https://magenta.tensorflow.org/datasets/maestro#v300
  与v200的差别就是去除了6个不属于钢琴音频的文件
  """

  df = pd.read_csv(f'{dataset_dir}/maestro-v3.0.0_tiny.csv', sep=',')
  wrong_files = [
    "2018/MIDI-Unprocessed_Chamber1_MID--AUDIO_07_R3_2018_wav--2",
    "2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--3",
    "2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--3",
    "2018/MIDI-Unprocessed_Chamber4_MID--AUDIO_11_R3_2018_wav--3",
    "2018/MIDI-Unprocessed_Chamber5_MID--AUDIO_18_R3_2018_wav--2",
    "2018/MIDI-Unprocessed_Chamber6_MID--AUDIO_20_R3_2018_wav--3",
  ]
  pattern = f"^{'|'.join(wrong_files)}"
  series = df['midi_filename'].str.contains(pattern, regex=True)
  idx = np.argwhere(series.values).flatten()
  df = df.drop(index=idx)
  df['audio_filename'] = df['audio_filename'].str.replace(r'\.wav$', '.mp3', regex=True)
  # 这里用的是kaggle上面12G的mp3数据集
  df.to_csv(f'{dataset_dir}/maestro-v3.0.0_tinymp3.csv', sep=',', index=False)
  logging.info('update metafile to v3.0.0')

  for name in wrong_files:
    midi, mp3 = f'{dataset_dir}/{name}.midi', f'{dataset_dir}/{name}.mp3'
    if os.path.isfile(midi):
      os.remove(midi)
      logging.info('remove midi file:', midi)
    if os.path.isfile(mp3):
      os.remove(mp3)
      logging.info('remove mp3 file:', mp3)


def read_metadata(csv_path, split=None):
    """Read metadata of MAESTRO dataset from csv file.

    Args:
      csv_path: str

    Returns:
      meta_dict, dict, e.g. {
        'canonical_composer': ['Alban Berg', ...], 
        'canonical_title': ['Sonata Op. 1', ...], 
        'split': ['train', ...], 
        'year': ['2018', ...]
        'midi_filename': ['2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi', ...], 
        'audio_filename': ['2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav', ...],
        'duration': [698.66116031, ...],
        'id': [2, 4, 6, ...]
      }
    """

    # meta_dict = {
    #   'canonical_composer': [], 'canonical_title': [], 'split': [], 
    #   'year': [], 'midi_filename': [], 'audio_filename': [], 'duration': []
    # }
    # with open(csv_path, 'r', encoding='utf-8') as f:
    #   csv_reader = csv.reader(f, delimiter=',')
    #   next(csv_reader)
    #   # 跳过第一行的表头
    #   for line in csv_reader:
    #     meta_dict['canonical_composer'].append(line[0])
    #     meta_dict['canonical_title'].append(line[1])
    #     meta_dict['split'].append(line[2])
    #     meta_dict['year'].append(line[3])
    #     meta_dict['midi_filename'].append(line[4])
    #     meta_dict['audio_filename'].append(line[5])
    #     meta_dict['duration'].append(float(line[6]))

    df = pd.read_csv(csv_path, delimiter=',')
    df['id'] = df.index
    if split is not None:
      df = df[df['split'] == split]

    return df.to_dict('list')


def _audio_to_frames(audio, config:BaseConfig):
  """将输入的音频数据切分成不重叠的帧和帧时长"""

  frame_size = config.FRAME_SIZE
  audio_len = len(audio)
  num_frames = np.ceil(audio_len / frame_size).astype(np.int32)
  # 将1-d的audio序列按frame_size为一帧切，看能切出多少帧

  samples = np.pad(audio, (0, num_frames*frame_size - audio_len), mode='constant')
  # 在末尾补0，便于下面切片；本能整除则不变
  logging.info(f'Padded {audio_len} samples to multiple of {frame_size}')

  # samples = np.asfortranarray(samples)
  # Fortran Order则指的是列优先的顺序
  frames = librosa.util.frame(samples, config.FRAME_SIZE, config.HOP_WIDTH, axis=0).astype(np.float32)
  logging.info(f'librosa.util.frame: {frames.shape=}')
  # 将samples沿着最后一维不重叠地切片；这里axis=0跟tf.signal.frame中-1效果一样
  # (5868, 128)

  # after _audio_to_frames, frames.shape = (5869, 128), frame_times.shape = (5869,) in dataset
  # 这是因为mt3在能整除的时候仍然pad一整份的frame_size，所以结果多了一个全0帧
  logging.info(f'Encoded {audio_len} samples to {num_frames} frames, {frame_size} samples each')

  # times = np.arange(num_frames, dtype=np.float32) / config.frames_per_second
  # return frames, times
  return frames


# features dict[str, tensor] -> examples sequence[features]
# 将读取的音频(1-d array)及midi切成不重叠的帧(2-d array)，处理sequence(midi序列)
def extract_features(
    audio: Sequence[np.float32],
    ns: note_seq.music_pb2.NoteSequence,
    duration: float,
    config: BaseConfig,
    codec: event_codec.Codec,
    include_ties: bool,
    example_id: str=None,
    onsets_only: bool=False,
) -> dict[str, Any]:
  """
  audio: librosa读出的mono、float的wav文件
  sequence：读取的midi文件
  """

  if example_id is not None:
    ns.id = example_id
    # 未赋值则为空

  logging.info(f'Got audio for {ns.id=}::{ns.filename=} with length {len(audio)}')
  frames = _audio_to_frames(audio, config)
  num_frames = np.ceil(duration*config.SAMPLE_RATE / config.FRAME_SIZE).astype(np.int32)
  frame_times = np.arange(num_frames, dtype=np.float32) / config.frames_per_second
  # 原本frame_times也在_audio_to_frames中计算，但现在一次只读出一个切片，不能靠audio的长度来计算
  # 这里num_frames以csv中记录的duration为准计算，因此与frames对应不上，TODO 后面或许可以先算出整个音频的features再存入h5待用

  if onsets_only:
    times, values = note_sequences.note_sequence_to_onsets(ns)
  else:
    # ns = note_seq.apply_sustain_control_changes(ns)
    # 将延音踏板cc事件通过更改ns的total_time混入了notes事件中，相当于提取了cc的信息到notes
    # TODO 可能有问题，处理过后延音线太多了
    # 踏板是有符号的，这里用加长音符时值的方式来近似让乐谱看起来杂乱，而且也不准确
    # 同理，乐谱还有许多事件，但都被忽略
    times, values = note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns)

  del ns.control_changes[:]
  # 这里最多的是CC#64，64号控制器被分配给了延音踏板（延音踏板的作用是使音符持续演奏，直至踏板抬起）。该控制器只有两个状态：开（数值大于64）和关（数值小于63)。
  # apply_sustain_control_changes中已经处理了cc，这里不再需要

  events, event_start_indices, event_end_indices, state_events, state_event_indices = \
    run_length_encoding.encode_and_index_events(
      state=note_sequences.NoteEncodingState() if include_ties else None,
      event_times=times,
      event_values=values,
      encode_event_fn=note_sequences.note_event_data_to_events,
      codec=codec,
      frame_times=frame_times,
      encoding_state_to_events_fn=note_sequences.note_encoding_state_to_events if include_ties else None
    )

  feature_to_trim = [event_start_indices, event_end_indices, state_event_indices]
  _, start_time = eval(example_id)
  start = int(start_time * config.SAMPLE_RATE / config.FRAME_SIZE)
  # end_time = min(start_time+config.segment_second, duration)
  # end = int(end_time * config.SAMPLE_RATE / config.FRAME_SIZE)
  end = start + config.MAX_INPUTS_LENGTH
  # SAMPLE_RATE 为16k，这里start, end都是无小数的浮点数，可以直接int转换
  event_start_indices, event_end_indices, state_event_indices = [x[start:end] for x in feature_to_trim]
  # 这里audio仅读取了一个片段，而其他特征对应的是完整的音频，需要在此根据audio将其裁剪
  logging.info(f'trim featrue: {start=}, {end=}, {start_time=}, {frame_times[start]=}')

  return {
    'inputs': frames,
    # 'input_times': frame_times,
    'targets': events,
    'input_event_start_indices': event_start_indices,
    'input_event_end_indices': event_end_indices,
    'input_state_event_indices': state_event_indices,
    'state_events': state_events,
    # 'sequence': ns.SerializeToString()
    # 这两个后续处理根本没用到
    # TODO input_state_event_indices似乎是全0，可能也能去掉
  }


def extract_features2(
    audio: Sequence[np.float32],
    ns: note_seq.music_pb2.NoteSequence,
    config: BaseConfig,
    codec: event_codec.Codec,
    start_time: float,
    end_time: float,
    example_id: str=None,
    onsets_only: bool=False,
) -> dict[str, Any]:
  """
  audio: librosa读出的mono、float的wav文件
  sequence：读取的midi文件
  """

  if example_id is not None:
    ns.id = example_id
    # 未赋值则为空

  logging.info(f'Got audio for {ns.id=}::{ns.filename=} with length {len(audio)}')
  frames = _audio_to_frames(audio, config)
  # num_frames = np.ceil(total_time*config.SAMPLE_RATE / config.FRAME_SIZE).astype(np.int32)

  if onsets_only:
    times, values = note_sequences.note_sequence_to_onsets(ns)
  else:
    # ns = note_seq.apply_sustain_control_changes(ns)
    # 将延音踏板cc事件通过更改ns的total_time混入了notes事件中，相当于提取了cc的信息到notes
    # TODO 可能有问题，处理过后延音线太多了
    # 踏板是有特定符号的，这里用加长音符时值的方式来近似让乐谱看起来杂乱，而且也不准确
    # 同理，乐谱还有许多事件，但这里都被忽略
    times, values = note_sequences.note_sequence_to_onsets_and_offsets(ns)

  events = run_length_encoding.encode_events(
    event_times=times,
    event_values=values,
    start_time=start_time,
    end_time=end_time,
    max_shift_steps=config.max_shift_steps,
    encode_event_fn=note_sequences.note_event_data_to_events,
    codec=codec,
    state_change_event_types=('velocity', )
  )

  # logging.info(f'{events=}')

  return {
    'inputs': frames,
    # 'input_times': frame_times,
    'targets': events,
  }


# 将tokenize的输出字典中部分feature再细切
def split_tokens(
  features,
  max_tokens_per_segment,
  min_tokens_per_segment = None,
  key = 'targets',
  additional_feature_keys = None,
  passthrough_feature_keys = None
) -> dict[str, Any]:
  """Split examples into multiple examples each

  Args:
    features: Dict of features to process.

  Returns:
    A dict of features.
  """
  # implementation of t5.data.preprocessors.split_tokens
  # in split_tokens, min_tokens_per_segment = None, max_tokens_per_segment = 512
  
  # features ==  dataset = <TensorDataset element_spec={
  #   'inputs': TensorSpec(shape=(5869, 128), dtype=tf.float32, name=None), 
  #   'input_times': TensorSpec(shape=(5869,), dtype=tf.float64, name=None)
  # }>

  if passthrough_feature_keys:
    split_keys = set([key] + (additional_feature_keys or []))
    overlap_keys = split_keys & set(passthrough_feature_keys)
    if overlap_keys:
      raise ValueError(f'split keys {overlap_keys} also included in passthrough keys')

  def _split_tokens(x):
    tokens = x[key]
    n_tokens = tokens.shape[0]

    if min_tokens_per_segment is None:
      length = np.int32(max_tokens_per_segment)
    else:
      # 根据对数均匀分布选择长度
      length = np.exp(np.random.randint(
          np.log(min_tokens_per_segment), 
          np.log(max_tokens_per_segment)
        )
      ).astype(np.int32)
      
    num_segments = np.ceil(n_tokens / length).astype(np.int32)
    padding = num_segments * length - n_tokens
    # 将tokens填充到length的整数倍再利用reshape将tokens切成num_segments段
    # 填充只为了reshape，之后还要把pad去掉

    feature_keys_to_split = [key]
    orig_lengths = {}
    outputs = {}
    if additional_feature_keys is not None:
      feature_keys_to_split.extend(additional_feature_keys)
    for k in feature_keys_to_split:
      assert x[k].shape[0]==n_tokens, f'Additional feature {k} is not the same size as {key} along axis 0 in split_tokens().'
      # 所有参与切分的特征必须在axis=0上大小相等
      
      # print(x[k].shape) # (5868, 128)
      shape = x[k].shape[1:]
      # padded1 = tf.pad(x[k],tf.concat([[[0, padding]],tf.zeros([len(shape_list), 2], dtype=tf.int32)],axis=0))
      # padded2 = np.pad(x[k], np.concatenate([[[0, padding]],torch.zeros([len(shape_list), 2], dtype=torch.int32)],axis=0))
      # padded3 = F.pad(x[k],tuple(torch.flatten(torch.cat([torch.tensor([[0, padding]], dtype=torch.int32), torch.zeros([len(shape_list), 2], dtype=torch.int32)], axis=0)).tolist()))
      padded = np.pad(x[k], np.concatenate(
          [[[0, padding]], np.zeros([len(shape), 2], dtype=np.int32)], axis=0)
      )
      # print(padded.shape) (6000, 128)
      # 将特征从5868帧，每帧128 增加到了6000帧，后面以帧为单位进行再切分

      orig_lengths[k] = np.concatenate([np.repeat(length, num_segments - 1), [length - padding]], axis=0).astype(np.int32)
      # 每个orig_lengths[k]也是array，记录切出来每一块的有效长度，如array([2000, 2000, 1868])
      outputs[k] = np.reshape(padded, np.concatenate([[-1, length], shape], axis=0).astype(np.int32))
      # 使用reashape将数据切成 (num_segments, length, frame_size)，如(3, 2000, 128)
      # print(outputs[k].shape)  # (3, 2000, 128)
      # 现在有3段，每段2000个帧，每帧长128；注意最后一段进行了pad，因此有效长度小于2000
      # max_length=512时: orig_lengths[k].shape) = TensorShape([12]), outputs[k].shape = TensorShape([12, 512, 128])

    if passthrough_feature_keys:
      for k in passthrough_feature_keys:
        outputs[k] = np.tile(
          np.expand_dims(x[k], axis=0),
          np.concatenate([[num_segments], np.tile([1], x[k].ndim)], axis=0)
        )                                # np.ones((x[k].ndim,), dtype=np.int32)

    logging.info(f'split_tokens._split_tokens, outputs:{get_feature_desc(outputs)}')
    logging.info(f'split_tokens._split_tokens, orig_lengths:{get_feature_desc(orig_lengths)}')
    return outputs, orig_lengths

  def _strip_padding(inputs, orig_lengths):
    output = {}
    for k, v in inputs.items():
      # 如这里v=(3, 2000, 128)
      if passthrough_feature_keys and k in passthrough_feature_keys:
        output[k] = v
      else:
        output[k] = []
        for i, x in enumerate(v):
          # (3, 2000, 128) 在第一维3上面循环
          output[k].append(x[:orig_lengths[k][i]])
          # output[k]=[(new_length, 128), (new_length, 128)...]
    return output

  # Filter empty examples.
  # dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[key]), 0))
  res = _split_tokens(features)
  res = _strip_padding(*res)
  # 现在res[inputs]=(12, ?, 128). 第二维前11都是512，最后一个是236
  logging.info(f'split_tokens, res:{get_feature_desc(res)}')
  return res


# 选择随机长度的tokens作为一块(chunk)
def select_random_chunk(
  features,
  min_length = None,
  max_length = 65536,
  key = 'targets',
  additional_feature_keys = None,
  passthrough_feature_keys = None,
  uniform_random_start = False
) -> dict[str, Any]:
  """Select a random chunk of tokens.

  Args:
    features: Dict of features to process.

  Returns:
    A dict of features.
  """
  # 根据t5.models.gin.objectives.denoise，preprocessors.select_random_chunk.max_length = 65536
  # 根据论文中 "The length of the selected segment can vary from a single input frame to the maximum input length" 
  # 结合 frame_length=HOP_WIDTH=128 猜测min_length=128

  if passthrough_feature_keys:
    chunk_keys = set([key] + (additional_feature_keys or []))
    overlap_keys = chunk_keys & set(passthrough_feature_keys)
    if overlap_keys:
      raise ValueError(f'chunk keys {overlap_keys} also included in passthrough keys')

  tokens = features[key]
  n_tokens = tokens.shape[0]


  if min_length is not None:
    length = np.random.randint(min_length, max_length)
  else:
    length = max_length

  if uniform_random_start:
    start = np.random.randint(-length + 1, n_tokens)
    end = min(start + length, n_tokens)
    start = max(start, 0)
  else:
    num_segments = np.ceil(n_tokens / length).astype(np.int32)
    start = length * np.random.randint(0, num_segments)
    end = min(start + length, n_tokens)

  chunk = {key: tokens[start:end]}
  logging.info(f'select_random_chunk, {key} {start=}, {end=}')

  if additional_feature_keys is not None:
    for k in additional_feature_keys:
      assert features[k].shape[0]==n_tokens, f'Additional feature {k} is not the same size as{key} along axis 0 in select_random_chunk().'
      chunk[k] = features[k][start:end]
      logging.info(f'select_random_chunk, {key} {start=}, {end=}')
  if passthrough_feature_keys is not None:
    for k in passthrough_feature_keys:
      chunk[k] = features[k]

  # 截取了原features中的一段，但类型、格式都没变
  # 长度根据模型输入而定，保证能遍历整个音频即可，最终chunk.shape=(segments, none, 128)
  return chunk


# 根据audio token片段抽取target
def extract_target_sequence_with_indices(
  features, 
  state_events_end_token=None
) -> dict[str, Any]:  
  """Extract target sequence corresponding to audio token segment.

  Args:
    features: Dict of features to process.

  Returns:
    A dict of features.
  """

  target_start_idx = features['input_event_start_indices'][0]
  target_end_idx = features['input_event_end_indices'][-1]
  # 这两个特征每一段都是1-d，如(1000,)与inputs的(1000, 128)呼应
  # 因此对于inputs(1000, 128)，本质跨越了1000帧。start,end也得相应取第一个跟最后一个元素才能包含这一段

  logging.info(f'{features["targets"].shape=}')
  features['targets'] = features['targets'][target_start_idx:target_end_idx]
  logging.info(f'{features["targets"].shape=}, {target_start_idx=} {target_end_idx=}')

  if state_events_end_token is not None:
    # Extract the state events corresponding to the audio start token, and
    # prepend them to the targets array.
    start_idx = features['input_state_event_indices'][0]
    end_idx = start_idx + 1
    while features['state_events'][end_idx - 1] != state_events_end_token:
      end_idx += 1
      features['targets'] = np.concatenate(
        [features['state_events'][start_idx:end_idx], features['targets']], axis=0
      )

  # logging.info(f'{features["targets"]=}')
  # inputs, shape=(512, 128); targets, shape=(442,); 
  # 丢弃input_event_start_indices; input_event_end_indices; input_state_event_indices; state_events
  return {
    'inputs': features['inputs'],
    'targets': features['targets']
  }


# 将midi program应用于token序列
# program是跟pitch、velocity同级的事件
def map_midi_programs(
    features,
    codec: event_codec.Codec,
    granularity_type: str = 'flat',
    key = 'targets'
) -> dict[str, Any]:
  """Apply MIDI program map to token sequences.
  
  Args:
    features: Dict of features to process.

  Returns:
    A dict of features.
  """

  granularity = vocabularies.PROGRAM_GRANULARITIES[granularity_type]
  # TODO 实际上根据mt3.gin.ismir2021，这里只会是flat，可以考虑去掉其他方式并整合
  logging.info(f'{key}, {features[key].shape=}')
  features[key] = granularity.tokens_map_fn(features[key], codec)
  logging.info(f'{key}, {features[key].shape=}')
  return features


# 使用行程编码压缩targets（只处理状态改变事件）
def run_length_encode_shifts_fn(
  features,
  codec: event_codec.Codec,
  key = 'targets',
  state_change_event_types = ()
) -> dict[str, Any]:
  """run-length encodes shifts for a given codec.
    Combine leading/interior shifts, trim trailing shifts.
    将之前run_length_encoding.encode_and_index_events生成的shift(index=1)事件压缩，末尾的shift直接去掉
    比如87个连续的shift(index=1)事件，这里就变成 "87" 这个事件index，大大节约了空间
    压缩比跟曲子的音符密度有关，越密集的曲子越难压缩（音符之间的间隔短）

    Args:
      features: Dict of features to process.

    Returns:
      A dict of features.
  """

  state_change_event_ranges = [codec.event_type_range(event_type) for event_type in state_change_event_types]
  # 获取state_change_event_types事件的编码起止点，每个事件范围都用个二元组表示
  events = features[key]
  logging.info(f'{len(state_change_event_ranges)=}, {state_change_event_ranges=}')
  logging.info(f'before RLE, {features[key].shape=}')
  logging.info(f'{features[key]=}')

  shift_steps = 0
  total_shift_steps = 0
  new_events = np.empty((0,), np.int32)  # == np.asarray([])
  current_state = np.zeros(len(state_change_event_ranges), dtype=np.int32)

  for event in events:
    if codec.is_shift_event_index(event):
      shift_steps += 1
      total_shift_steps += 1
      # 属于shift_event，累计steps
      # 不属于才下面的else输出，因此会发生new_events第一个元素为steps的事情

    else:
      # 遇到state change事件，且目标状态与当前状态相同则跳过
      is_redundant = False
      for i, (min_index, max_index) in enumerate(state_change_event_ranges):
        if min_index <= event <= max_index:
          if current_state[i] == event:
            # 连续多个音符结束时一直保持velocity=0即可，故也可省略
            is_redundant = True
            # 当前已经在这个状态上，则忽略该状态转换事件
          current_state[i] = event

      if is_redundant:
        continue

      # 遇到新状态改变，使用行程编码记录前一个事件的偏移
      if shift_steps > 0:
        shift_steps = total_shift_steps
        while shift_steps > 0:
          new_events_steps = min(codec.max_shift_steps, shift_steps)
          new_events = np.append(new_events, new_events_steps)
          # logging.info(f"RLE, add {new_events_steps=}")
          shift_steps -= new_events_steps
          # 跟一般的RLE不一样，这里记录的是距离起点的绝对偏移total_shift_steps不断在累计
          # 且有最大行程限制，若连续的shift事件超过max_shift_steps就先生成一个事件，剩下的另行生成事件
          # 因为max_shift_steps往后的数字分配给了 velocity, pitch 等事件，不可用于shift
          # 但实际上RLE这里的target只是一个片段，每个片段独立计数，total_steps不会太大
      new_events = np.append(new_events, event)
      # logging.info(f"RLE, add {event=}")

  features[key] = new_events
  logging.info(f'after RLE, {features[key].shape=}')
  # logging.info(f'{features[key]=}')
  return features


# 计算对数梅尔频谱图
def compute_spectrograms(
  features,
  config: BaseConfig
) -> dict[str, Any]:
  samples = np.reshape(features['inputs'], (-1,))  
  logging.info(f'{samples.shape=}')  # samples.shape=(131072,)

  mel_spec = librosa.feature.melspectrogram(
    samples, sr=config.SAMPLE_RATE, n_fft=config.FFT_SIZE, 
    hop_length=config.HOP_WIDTH, win_length=config.FFT_SIZE,
    window='hann', center=True, pad_mode='reflect', n_mels=config.NUM_MEL_BINS, 
    fmin=config.MEL_LO_HZ, fmax=config.MEL_HI_HZ  #, norm=1  # 将三角mel权重除以mel带的宽度（区域归一化）
  )
  # center, hop_length参数影响第一维，n_mels决定第二维
  log_mel_spec = librosa.power_to_db(mel_spec)  # log_mel_spec.shape=(512, 1025)

  # 对数梅尔频谱的计算：https://zhuanlan.zhihu.com/p/350846654，https://zhuanlan.zhihu.com/p/351956040
  # from torchaudio.transforms import MelSpectrogram
  # mel_spectrogram = MelSpectrogram(
  #   sample_rate = cf.SAMPLE_RATE,
  #   n_fft = cf.FFT_SIZE,
  #   win_length = cf.FFT_SIZE,
  #   hop_length = cf.HOP_WIDTH,
  #   f_min = cf.MEL_LO_HZ,
  #   f_max = cf.MEL_HI_HZ,
  #   n_mels = cf.NUM_MEL_BINS,
  #   center = True,
  #   pad_mode = 'reflect'
  # )
  # log_mel_spectrogram = 10.0 * torch.log10(torch.clamp(mel_spectrogram, min=1e-10, max=np.inf))
  # log_mel_spectrogram -= 10.0 * np.log10(1.0)

  log_mel_spec = log_mel_spec.T[:-1]
  # 丢掉最后一维，使(512, 1025) -> (1024, 512)
  logging.info(f'{log_mel_spec.shape=}')
  features['inputs'] = log_mel_spec
  # features['raw_inputs'] = samples
  return features


# 将特征targets进行tokenize，并加上EOS
def tokenize(
  features: Mapping[str, Any],
  vocab:vocabularies.GenericTokenVocabulary,
  key: str = 'targets',
  with_eos: bool = False
) -> dict[str, Any]:
  """Encode output features with specified vocbularies and append EOS.

  When `with_eos` is True and input features are ranked > 1, then an EOS is
  appended only to the last item of each 1-D sequence.
  """

  v = features[key]

  logging.info(f"before tokenize, {v[:20]=}, {v[-20:]}")
  v = vocab.encode(v) # v: a list of integers, (n,)
  # 将所有元素加上3，为3个特殊符号留位置
  v = np.asarray(v)
  assert v.ndim == 1, f"wrong {v.ndim=} of feature {key}'s values"

  if with_eos:
    v = np.append(v, vocab.eos_id)
  logging.info(f"after tokenize, {v[:20]=}, {v[-20:]}")

  features[key] = v
  return features


# 将抽取的特征转换成适合模型的输入
def convert_features(
  features: Mapping[str, Any],
  config: BaseConfig
) -> dict[str, Any]:
  # inputs shape=(512, 512), targets shape=(83,); 
  def max_length_for_key(key):
    max_length = getattr(config, f"MAX_{key.upper()}_LENGTH", -1)
    return max_length

  for k in {'inputs', 'targets'}:
    if k not in features:
      logging.exception(f'key "{k}" does not exits in features')
      exit(-1)
  
    v =  features[k]
    v_len = v.shape[0]
    if v_len > (max_v_len := max_length_for_key(k)):
      logging.exception(f'{v_len=} for "{k}" field exceeds maximum length {max_v_len}')
      exit(-1)
      # 特征不能裁剪，只能限制不超出长度
    
    features[k] = np.pad(features[k], [(0, max_v_len-v_len)] + [(0, 0)]*(v.ndim - 1), mode='constant')
    # 只pad第一维，其他维度都不进行pad
  
  decoder_input_tokens = np.concatenate(([0], features["targets"][:-1]), axis=0)

  return {
    "encoder_input_tokens": features["inputs"],
    "decoder_target_tokens": features["targets"],
    "decoder_input_tokens": decoder_input_tokens,
    # 去掉最后一个元素并在最前方插入起始符0作为decoder输入，最后一个元素要么pad的0要么eos，没有影响
  }

def main(cf: BaseConfig):
    start_time = time.time()
    logs_dir = os.path.join(cf.WORKSPACE, 'logs')
    create_logging(logs_dir, filemode='w')

    # csv_path = os.path.join(cf.DATASET_DIR, 'maestro-v3.0.0_tiny.csv')
    # # Read meta dict
    # meta_dict = read_metadata(csv_path)
    # # 实际上dataframe不换成dict似乎也可以

    # # Read as note_sequence
    # sample_id = 0
    # midi_path = os.path.join(cf.DATASET_DIR, meta_dict['midi_filename'][sample_id])
    # note_sequence = note_seq.midi_file_to_note_sequence(midi_path)
    # # Load audio
    # audio_path = os.path.join(cf.DATASET_DIR, meta_dict['audio_filename'][sample_id])
    # audio, _ = librosa.core.load(audio_path, sr=cf.SAMPLE_RATE, mono=True)

    # codec = vocabularies.build_codec(cf)
    # # vocabulary = vocabularies.vocabulary_from_codec(codec)
    # f = extract_features(audio, note_sequence, cf, include_ties=False, codec=codec, example_id=str(sample_id))
    # # np.savez("./cache/extract_features.npz", **f)

    # f = np.load("./cache/extract_features.npz", allow_pickle=True)
    # TODO 这函数最后strip_pad，特征变为装了array的list，或许不去掉pad也可
    # f = split_tokens(
    #   f,
    #   max_tokens_per_segment=cf.MAX_SEGMENT_LENGTH,
    #   key='inputs',
    #   additional_feature_keys=[
    #       'input_event_start_indices', 'input_event_end_indices',
    #       'input_state_event_indices'
    #   ],
    #   passthrough_feature_keys=['targets', 'state_events']
    # )
    # np.savez("./cache/split_tokens.npz", **f)

    f = np.load("./cache/split_tokens.npz", allow_pickle=True)
    # TODO 这里从每段(长2000)中抽一部分参与训练，后面应该放到sample里
    f = select_random_chunk(
      f,
      min_length = cf.MAX_INPUT_LENGTH,
      max_length = cf.MAX_SEGMENT_LENGTH,
      key='inputs',
      additional_feature_keys=[
          'input_event_start_indices', 'input_event_end_indices',
          'input_state_event_indices'
      ],
      passthrough_feature_keys=['targets', 'state_events'],
      uniform_random_start=True
    )
    np.savez("./cache/select_random_chunk.npz", **f)

    print(f'Time: {time.time() - start_time:.3f} s')

if __name__ == '__main__':
  config = DevConfig()

  # pack_maestro_dataset_to_hdf5(args)
  main(config)

