import os
import time
import logging

import numpy as np
import torch
import torch.nn.functional as F
import tensorflow as tf
import h5py
import csv
import librosa
from mido import MidiFile
import note_seq

from yui import event_codec
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


def _audio_to_frames(samples, spectrogram_config:BaseConfig):
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
  # TODO 用torchlibrosa.stft.Spectrogram替换

  num_frames = len(samples) // frame_size
  logging.info('Encoded %d samples to %d frames (%d samples each)', len(samples), num_frames, frame_size)

  times = np.arange(num_frames) / spectrogram_config.frames_per_second
  return frames, times



# TODO 后续可能整合到Smapler中，输入输出都应改成batch，目前都是处理单个数据
# 将读取的音频及midi切成不重叠的帧
def tokenize(
    samples, sequence, spectrogram_config,
    codec, example_id=None,
    onsets_only=False, include_ties=False, audio_is_samples=False,
    id_feature_key='id'
):

  """
  samples: librosa读出的mono、float的wav文件
  sequence：读取的midi文件
  """

  if onsets_only and include_ties:
    raise ValueError('Ties not supported when only modeling onsets.')

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
  data,
  max_tokens_per_segment,
  num_parallel_calls,
  min_tokens_per_segment = None,
  feature_key = 'targets',
  additional_feature_keys = None,
  passthrough_feature_keys = None,
  **unused_kwargs
):
  # 来自t5.data.preprocessors.split_tokens

  if passthrough_feature_keys:
    split_keys = set([feature_key] + (additional_feature_keys or []))
    overlap_keys = split_keys & set(passthrough_feature_keys)
    if overlap_keys:
      raise ValueError(f'split keys {overlap_keys} also included in passthrough keys')

  def _split_tokens(x):
    """Split one token sequence into multiple sequences."""

    tokens = x[feature_key]
    # n_tokens = tf.shape(tokens)[0]
    n_tokens = tokens.shape[0]
    length = max_tokens_per_segment
    # 对于maetro总是最大长度切片，故省略对数均匀分布的实现

    num_segments = torch.ceil(n_tokens / length)
    padding = num_segments * length - n_tokens
    # 将tokens填充到能整除length的长度再切片

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
  res = _split_tokens(data)
  res = _strip_padding(*res)
  return res


def select_random_chunk(dataset: tf.data.Dataset,
                        output_features: Mapping[str, seqio.Feature],
                        max_length: Optional[int] = None,
                        feature_key: str = 'targets',
                        additional_feature_keys: Optional[Sequence[str]] = None,
                        passthrough_feature_keys: Optional[
                            Sequence[str]] = None,
                        sequence_length: Optional[Mapping[str, int]] = None,
                        uniform_random_start: bool = False,
                        min_length: Optional[int] = None,
                        **unused_kwargs) -> tf.data.Dataset:
  """Token-preprocessor to extract one span of at most `max_length` tokens.

  If the token sequence is longer than `max_length`, then we return a random
  subsequence.  Otherwise, we return the full sequence.

  This is generally followed by split_tokens.

  Args:
    dataset: A tf.data.Dataset with dictionaries containing the key feature_key.
    output_features: Mapping of keys to features.
    max_length: Typically specified in gin configs, takes priority over
      sequence_length.
    feature_key: Which feature to use from the dataset.
    additional_feature_keys: Additional features to use. The same chunk will be
      selected from these features as from the one specified in feature_key,
      so they should all have the same length.
    passthrough_feature_keys: Additional keys to pass through unchanged.
    sequence_length: Used if max_length is not specified. Typically passed in
      by the data pipeline. feature_key will be used to select the length.
    uniform_random_start: If True, will select a starting point in
      [-max_length + 1, n_tokens). If False, will select one of a set of chunks
      offset by max_length. Both of these starting points try to ensure each
      token has an equal probability of being included.
    min_length: If specified, lengths of chunks will be selected uniformly at
      random from [min_length, max_length]. Note that chunks can end up shorter
      than min_length if at the beginning or end of the sequence.

  Returns:
    a dataset
  """
  if passthrough_feature_keys:
    chunk_keys = set([feature_key] + (additional_feature_keys or []))
    overlap_keys = chunk_keys & set(passthrough_feature_keys)
    if overlap_keys:
      raise ValueError(
          f'chunk keys {overlap_keys} also included in passthrough keys')

  if max_length is None and sequence_length is not None:
    max_length = sequence_length[feature_key]
    if output_features[feature_key].add_eos:
      # Leave room to insert an EOS token.
      max_length -= 1
  if max_length is None:
    raise ValueError('Must specify max_length or sequence_length.')

  @seqio.map_over_dataset(num_seeds=2)
  def _my_fn(x, seeds):
    """Select a random chunk of tokens.

    Args:
      x: a 1d Tensor
      seeds: an int32 Tensor, shaped (2, 2), the random seeds.
    Returns:
      a 1d Tensor
    """
    tokens = x[feature_key]
    n_tokens = tf.shape(tokens)[0]
    if min_length is not None:
      length = tf.random.stateless_uniform(
          [],
          minval=min_length,
          maxval=max_length,
          dtype=tf.int32,
          seed=seeds[0])
    else:
      length = max_length
    if uniform_random_start:
      start = tf.random.stateless_uniform(
          [],
          minval=-length + 1,  # pylint:disable=invalid-unary-operand-type
          maxval=n_tokens,
          dtype=tf.int32,
          seed=seeds[1])
      end = tf.minimum(start + length, n_tokens)
      start = tf.maximum(start, 0)
    else:
      num_segments = tf.cast(
          tf.math.ceil(
              tf.cast(n_tokens, tf.float32) / tf.cast(length, tf.float32)
          ),
          tf.int32)
      start = length * tf.random.stateless_uniform(
          [],
          maxval=num_segments,
          dtype=tf.int32,
          seed=seeds[1])
      end = tf.minimum(start + length, n_tokens)
    chunk = {feature_key: tokens[start:end]}
    if additional_feature_keys is not None:
      for k in additional_feature_keys:
        with tf.control_dependencies([
            tf.assert_equal(
                tf.shape(tokens)[0],
                tf.shape(x[k])[0],
                message=(f'Additional feature {k} is not the same size as '
                         f'{feature_key} along axis 0 in select_random_chunk().'
                         )
            )
        ]):
          chunk[k] = x[k][start:end]
    if passthrough_feature_keys is not None:
      for k in passthrough_feature_keys:
        chunk[k] = x[k]
    return chunk
  # Filter empty examples.
  dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[feature_key]), 0))
  return _my_fn(dataset)


if __name__ == '__main__':
    args = Namespace(
        dataset_dir=cf.DATASET_DIR,
        workspace=cf.WORKSPACE,
    )
    # Directory of dataset
    # Directory of your workspace

    pack_maestro_dataset_to_hdf5(args)
