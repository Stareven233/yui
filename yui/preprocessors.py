import imp
import os
import time
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import tensorflow as tf
import h5py
import csv
import librosa
from mido import MidiFile
import note_seq

from yui import event_codec, vocabularies
from yui import note_sequences
from yui import run_length_encoding
from yui.utils import Namespace, create_folder, get_filename, create_logging, float32_to_int16
from config.data import BaseConfig, DevConfig


cf = DevConfig()


def read_metadata(csv_path):
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
        'duration': [698.66116031, ...]}
    """

    with open(csv_path, 'r', encoding='utf-8') as fr:
        reader = csv.reader(fr, delimiter=',')
        lines = list(reader)

    meta_dict = {'canonical_composer': [], 'canonical_title': [], 'split': [], 
        'year': [], 'midi_filename': [], 'audio_filename': [], 'duration': []}

    for n in range(1, len(lines)):
        meta_dict['canonical_composer'].append(lines[n][0])
        meta_dict['canonical_title'].append(lines[n][1])
        meta_dict['split'].append(lines[n][2])
        meta_dict['year'].append(lines[n][3])
        meta_dict['midi_filename'].append(lines[n][4])
        meta_dict['audio_filename'].append(lines[n][5])
        meta_dict['duration'].append(float(lines[n][6]))
    # TODO 感觉这段相当值得优化，尝试pandas?

    for key in meta_dict.keys():
        meta_dict[key] = np.array(meta_dict[key])
    
    return meta_dict


def read_midi(midi_path):
    """Parse MIDI file.

    Args:
      midi_path: str

    Returns:
      midi_dict: dict, e.g. {
        'midi_event': [
            'program_change channel=0 program=0 time=0', 
            'control_change channel=0 control=64 value=127 time=0', 
            'control_change channel=0 control=64 value=63 time=236', 
            ...],
        'midi_event_time': [0., 0, 0.98307292, ...]}
    """

    # TODO 可能需要修改成输出MT3那样的类MIDI事件

    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat

    assert len(midi_file.tracks) == 2
    """The first track contains tempo, time signature. The second track 
    contains piano events."""

    microseconds_per_beat = midi_file.tracks[0][0].tempo
    beats_per_second = 1e6 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second

    message_list = []

    ticks = 0
    time_in_second = []

    for message in midi_file.tracks[1]:
        message_list.append(str(message))
        ticks += message.time
        time_in_second.append(ticks / ticks_per_second)

    midi_dict = {
        'midi_event': np.array(message_list), 
        'midi_event_time': np.array(time_in_second)}

    return midi_dict


def pack_maestro_dataset_to_hdf5(args):
    """Load & resample MAESTRO audio files, then write to hdf5 files.

    Args:
      dataset_dir: str, directory of dataset
      workspace: str, directory of your workspace
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    sample_rate = cf.SAMPLE_RATE

    # Paths
    csv_path = os.path.join(dataset_dir, 'maestro-v3.0.0_tiny.csv')
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'maestro')

    logs_dir = os.path.join(workspace, 'logs', get_filename(__file__))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    # Read meta dict
    meta_dict = read_metadata(csv_path)

    audios_num = len(meta_dict['canonical_composer'])
    logging.info('Total audios number: {}'.format(audios_num))

    feature_time = time.time()

    # Load & resample each audio file to a hdf5 file
    for n in range(audios_num):
        logging.info('{} {}'.format(n, meta_dict['midi_filename'][n]))

        # Read midi
        midi_path = os.path.join(dataset_dir, meta_dict['midi_filename'][n])
        midi_dict = read_midi(midi_path)

        # Load audio
        audio_path = os.path.join(dataset_dir, meta_dict['audio_filename'][n])
        (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
        # 跟mt3依赖同一个函数：librosa.load, 读取的audio也一致

        packed_hdf5_path = os.path.join(waveform_hdf5s_dir, '{}.h5'.format(
            os.path.splitext(meta_dict['audio_filename'][n])[0]
        ))
        create_folder(os.path.dirname(packed_hdf5_path))

        with h5py.File(packed_hdf5_path, 'w') as hf:
            hf.attrs.create('canonical_composer', data=meta_dict['canonical_composer'][n].encode(), dtype='S100')
            hf.attrs.create('canonical_title', data=meta_dict['canonical_title'][n].encode(), dtype='S100')
            hf.attrs.create('split', data=meta_dict['split'][n].encode(), dtype='S20')
            hf.attrs.create('year', data=meta_dict['year'][n].encode(), dtype='S10')
            hf.attrs.create('midi_filename', data=meta_dict['midi_filename'][n].encode(), dtype='S100')
            hf.attrs.create('audio_filename', data=meta_dict['audio_filename'][n].encode(), dtype='S100')
            hf.attrs.create('duration', data=meta_dict['duration'][n], dtype=np.float32)

            hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100')
            hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32)
            hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
        
    logging.info(f'Write hdf5 to {waveform_hdf5s_dir}')
    logging.info(f'Time: {time.time() - feature_time:.3f} s')


def _audio_to_frames(samples, spectrogram_config):
  """将输入的音频数据切分成不重叠的帧和帧时长"""

  frame_size = spectrogram_config.HOP_WIDTH
  samples = np.pad(samples, [0, frame_size - len(samples) % frame_size], mode='constant')
  logging.info('Padded %d samples to multiple of %d', len(samples), frame_size)

  frames = tf.signal.frame(
      samples,
      frame_length=spectrogram_config.HOP_WIDTH,
      frame_step=spectrogram_config.HOP_WIDTH,
      pad_end=True)
  #  不重叠地切片.
  # TODO 用torch/np其换掉tf函数

  num_frames = len(samples) // frame_size
  logging.info('Encoded %d samples to %d frames (%d samples each)', len(samples), num_frames, frame_size)

  times = np.arange(num_frames) / spectrogram_config.frames_per_second
  return frames, times


# TODO 后续可能整合到Smapler中，输入输出都应改成batch，目前都是处理单个数据
# 将读取的音频及midi切成不重叠的帧
def tokenize(
    samples, sequence, spectrogram_config,
    codec, example_id=None,
    onsets_only=False,
    id_feature_key='id'
):
  """
  samples: librosa读出的mono、float的wav文件
  sequence：读取的midi文件
  """

  ns = note_seq.NoteSequence.FromString(sequence)
  note_sequences.validate_note_sequence(ns)

  if example_id is not None:
    ns.id = example_id
    
  logging.info('Got samples for %s::%s with length %d', ns.id, ns.filename, len(samples))

  frames, frame_times = _audio_to_frames(samples, spectrogram_config)

  if onsets_only:
    times, values = note_sequences.note_sequence_to_onsets(ns)
  else:
    ns = note_seq.apply_sustain_control_changes(ns)
    times, values = (note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns))

  # The original NoteSequence can have a lot of control changes we don't need;
  # delete them.
  del ns.control_changes[:]

  (events, event_start_indices, event_end_indices,
    state_events, state_event_indices) = (
      run_length_encoding.encode_and_index_events(
          state=note_sequences.NoteEncodingState() if include_ties else None,
          event_times=times,
          event_values=values,
          encode_event_fn=note_sequences.note_event_data_to_events,
          codec=codec,
          frame_times=frame_times,
          encoding_state_to_events_fn=(
              note_sequences.note_encoding_state_to_events
              if include_ties else None)
          )
      )

  return {
    'inputs': frames,
    'input_times': frame_times,
    'targets': events,
    'input_event_start_indices': event_start_indices,
    'input_event_end_indices': event_end_indices,
    'state_events': state_events,
    'input_state_event_indices': state_event_indices,
    'sequence': ns.SerializeToString()
  }


# 将tokenize的输出字典中部分feature再细切
def split_tokens(
  features,
  max_tokens_per_segment,
  min_tokens_per_segment = None,
  feature_key = 'targets',
  additional_feature_keys = None,
  passthrough_feature_keys = None
):
  """Split examples into multiple examples each

  Args:
    features: Dict of features to process.

  Returns:
    A dict of features.
  """
  # 来自t5.data.preprocessors.split_tokens

  if passthrough_feature_keys:
    split_keys = set([feature_key] + (additional_feature_keys or []))
    overlap_keys = split_keys & set(passthrough_feature_keys)
    if overlap_keys:
      raise ValueError(f'split keys {overlap_keys} also included in passthrough keys')

  def _split_tokens(x):
    tokens = x[feature_key]
    n_tokens = tokens.shape[0]

    if min_tokens_per_segment is None:
      length = max_tokens_per_segment
    else:
      # 根据对数均匀分布选择长度
      length = np.exp(np.random.randint(
          np.log(min_tokens_per_segment), 
          np.log(max_tokens_per_segment)
        )
      ).astype(np.int32)
      
    num_segments = torch.ceil(n_tokens / length)
    padding = num_segments * length - n_tokens
    # 将tokens填充到length的整数倍再利用reshape将tokens切成num_segments段

    feature_keys_to_split = [feature_key]
    orig_lengths = {}
    outputs = {}
    if additional_feature_keys is not None:
      feature_keys_to_split.extend(additional_feature_keys)
    for k in feature_keys_to_split:
      assert x[k].shape[0]==n_tokens, f'Additional feature {k} is not the same size as{feature_key} along axis 0 in split_tokens().'
      # 所有参与切分的特征必须在axis=0上大小相等
      
      shape = x[k].shape[1:]
      # padded1 = tf.pad(x[k],tf.concat([[[0, padding]],tf.zeros([len(shape_list), 2], dtype=tf.int32)],axis=0))
      # padded2 = np.pad(x[k], np.concatenate([[[0, padding]],torch.zeros([len(shape_list), 2], dtype=torch.int32)],axis=0))
      # padded3 = F.pad(x[k],tuple(torch.flatten(torch.cat([torch.tensor([[0, padding]], dtype=torch.int32), torch.zeros([len(shape_list), 2], dtype=torch.int32)], axis=0)).tolist()))
      padded = np.pad(x[k], np.concatenate(
          [[[0, padding]], np.zeros([len(shape), 2], dtype=np.int32)], axis=0)
      )
      # TODO 同名函数中torch行为与tensorflow不同，此处都用numpy的函数代替，
      # 可能也会有bug，到时候需检查是否将结果再换回torch.Tensor...

      orig_lengths[k] = np.concatenate([np.repeat(length, num_segments - 1), [length - padding]], axis=0)
      outputs[k] = np.reshape(padded, np.concatenate([[-1, length], shape], axis=0))
      # 使用reashape将数据切成(batch, length, shape)？

    if passthrough_feature_keys:
      for k in passthrough_feature_keys:
        outputs[k] = np.tile(
          np.expand_dims(x[k], axis=0),
          np.concatenate([[num_segments], tf.tile([1], [tf.rank(x[k])])], axis=0)
        )
    return outputs, orig_lengths

  def _strip_padding(inputs, orig_lengths):
    output = {}
    for k, v in inputs.items():
      if passthrough_feature_keys and k in passthrough_feature_keys:
        output[k] = v
      else:
        output[k] = v[:orig_lengths[k]]
    return output

  # Filter empty examples.
  # dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[feature_key]), 0))
  res = _split_tokens(features)
  res = _strip_padding(*res)
  assert type(res) == dict, "res in preprocessors.split_tokens is not a dict"
  return res


# 选择随机长度的tokens作为一块(chunk)
def select_random_chunk(
  features,
  min_length = 128,
  max_length = 65536,
  feature_key = 'targets',
  additional_feature_keys = None,
  passthrough_feature_keys = None,
  uniform_random_start = False
):
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
    chunk_keys = set([feature_key] + (additional_feature_keys or []))
    overlap_keys = chunk_keys & set(passthrough_feature_keys)
    if overlap_keys:
      raise ValueError(f'chunk keys {overlap_keys} also included in passthrough keys')

  tokens = features[feature_key]
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

  chunk = {feature_key: tokens[start:end]}
  if additional_feature_keys is not None:
    for k in additional_feature_keys:
      assert features[k].shape[0]==n_tokens, f'Additional feature {k} is not the same size as{feature_key} along axis 0 in select_random_chunk().'
      chunk[k] = features[k][start:end]
  if passthrough_feature_keys is not None:
    for k in passthrough_feature_keys:
      chunk[k] = features[k]

  # 截取了原features中的一段，但类型、格式都没变
  return chunk


# 根据audio token片段抽取target
def extract_target_sequence_with_indices(features, state_events_end_token=None):  
  """Extract target sequence corresponding to audio token segment.

  Args:
    features: Dict of features to process.

  Returns:
    A dict of features.
  """

  target_start_idx = features['input_event_start_indices'][0]
  target_end_idx = features['input_event_end_indices'][-1]

  features['targets'] = features['targets'][target_start_idx:target_end_idx]

  if state_events_end_token is not None:
    # Extract the state events corresponding to the audio start token, and
    # prepend them to the targets array.
    start_idx = features['input_state_event_indices'][0]
    end_idx = start_idx + 1
    while features['state_events'][end_idx - 1] != state_events_end_token:
      end_idx += 1
    features['targets'] = np.concatenate([
      features['state_events'][start_idx:end_idx],
      features['targets']
    ], axis=0)

  return features


# 将midi program应用于token序列
# program是跟pitch、velocity同级的事件?
def map_midi_programs(
    features,
    codec: event_codec.Codec,
    granularity_type: str = 'full',
    feature_key = 'targets'
):
  """Apply MIDI program map to token sequences.
  
  Args:
    features: Dict of features to process.

  Returns:
    A dict of features.
  """

  granularity = vocabularies.PROGRAM_GRANULARITIES[granularity_type]
  # TODO 实际上根据mt3.gin.ismir2021，这里只会是flat，可以考虑去掉其他方式并整合

  features[feature_key] = granularity.tokens_map_fn(features[feature_key], codec)
  return features


# 处理类midi事件，具体不明
def run_length_encode_shifts_fn(
  features,
  codec: event_codec.Codec,
  feature_key = 'targets',
  state_change_event_types = ()
):
  """run-length encodes shifts for a given codec.
    Combine leading/interior shifts, trim trailing shifts.

    Args:
      features: Dict of features to process.

    Returns:
      A dict of features.
  """

  state_change_event_ranges = [codec.event_type_range(event_type)
                               for event_type in state_change_event_types]
  events = features[feature_key]

  shift_steps = 0
  total_shift_steps = 0
  output = np.empty((0,), np.int32)  # == np.asarray([])
  current_state = np.zeros(len(state_change_event_ranges), dtype=np.int32)

  for event in events:
    if codec.is_shift_event_index(event):
      shift_steps += 1
      total_shift_steps += 1

    else:
      # If this event is a state change and has the same value as the current
      # state, we can skip it entirely.
      is_redundant = False
      for i, (min_index, max_index) in enumerate(state_change_event_ranges):
        if (min_index <= event) and (event <= max_index):
          if current_state[i] == event:
            is_redundant = True
          current_state[i] = event

      if is_redundant:
        continue
      # Once we've reached a non-shift event, RLE all previous shift events
      # before outputting the non-shift event.
      if shift_steps > 0:
        shift_steps = total_shift_steps
        while shift_steps > 0:
          output_steps = min(codec.max_shift_steps, shift_steps)
          output = np.concatenate([output, [output_steps]], axis=0)
          shift_steps -= output_steps
      output = np.concatenate([output, [event]], axis=0)

  features[feature_key] = output
  return features


# 计算对数梅尔频谱图
def compute_spectrograms(features, cf:BaseConfig):
  samples = np.reshape(features['inputs'], (-1,))
  spectrogram_extractor = Spectrogram(n_fft=cf.FFT_SIZE, 
      hop_length=cf.HOP_WIDTH, win_length=cf.FFT_SIZE, window='hann', 
      center=True, pad_mode='reflect', freeze_parameters=True
  )
  # Logmel feature extractor
  logmel_extractor = LogmelFilterBank(sr=cf.SAMPLE_RATE, 
      n_fft=cf.FFT_SIZE, n_mels=cf.NUM_MEL_BINS, fmin=cf.MEL_LO_HZ, fmax=cf.MEL_HI_HZ, ref=1.0, 
      amin=1e-10, top_db=None, freeze_parameters=True)
  # 对数梅尔频谱的计算：https://zhuanlan.zhihu.com/p/350846654，https://zhuanlan.zhihu.com/p/351956040

  # TODO 或许可以用https://pytorch.org/audio/stable/transforms.html#melspectrogram代替
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

  features['inputs'] = logmel_extractor(spectrogram_extractor(samples))
  features['raw_inputs'] = samples
  return features


def handle_too_long(
  features,
  output_features,
  sequence_length,
  skip = False
):
  """Handle sequences that are too long, by either failing or skipping them."""
  
  def max_length_for_key(key):
    max_length = sequence_length[key]
    if output_features[key].add_eos:
      max_length -= 1
    return max_length

  if skip:
    # Drop examples where one of the features is longer than its maximum
    # sequence length.
    def is_not_too_long(ex):
      return not tf.reduce_any(
          [k in output_features and len(v) > max_length_for_key(k)
           for k, v in ex.items()])
    dataset = dataset.filter(is_not_too_long)

  def assert_not_too_long(key: str, value: tf.Tensor) -> tf.Tensor:
    if key in output_features:
      max_length = max_length_for_key(key)
      tf.debugging.assert_less_equal(
          tf.shape(value)[0], max_length,
          f'Value for "{key}" field exceeds maximum length')
    return value

  # Assert that no examples have features longer than their maximum sequence
  # length.
  return dataset.map(
      lambda ex: {k: assert_not_too_long(k, v) for k, v in ex.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE)


if __name__ == '__main__':
    args = Namespace(
        dataset_dir=cf.DATASET_DIR,
        workspace=cf.WORKSPACE,
    )
    # Directory of dataset
    # Directory of your workspace

    pack_maestro_dataset_to_hdf5(args)
