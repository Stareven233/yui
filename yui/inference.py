"""inference
转录单文件音频
"""

import os
import time
import logging
import functools
from typing import Callable, Generator, Sequence
import argparse

import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Config
import note_seq
import librosa
import pretty_midi

import vocabularies
import config
from config.data import YuiConfig
import utils
import preprocessors
import postprocessors


def audio_split_to_batch(audio: str, batch_size: int, config: YuiConfig) -> Generator:
  """将待转录的一首曲子切片预处理，之后逐batch返回"""

  audio, _ = librosa.core.load(audio, sr=config.SAMPLE_RATE, mono=True)
  audio_len = len(audio)
  segment_len = int(config.segment_second * config.SAMPLE_RATE)
  sample_batch, mask_batch = [], []
  start_time = np.arange(0, audio_len, segment_len)
  start = 0
  segment_num = len(start_time)
  i = 1
  while i <= segment_num:
  # for i in range(1, segment_num+1):
    start = start_time[i-1]
    end = audio_len if i==segment_num else start_time[i]
    sample = audio[start:end]
    
    sample = preprocessors.audio_to_frames(sample, config)
    sample = preprocessors.compute_spectrograms({'inputs': sample}, config).get('inputs')
    inputs_len = sample.shape[0]
    mask = np.ones((config.MAX_INPUTS_LENGTH, ), dtype=np.bool8)
    mask[inputs_len:] = 0
    sample = np.pad(sample, ((0, config.MAX_INPUTS_LENGTH-inputs_len), (0, 0)), constant_values=config.PAD_ID)  # (v_len, mel_bins)
    
    sample_batch.append(sample)
    mask_batch.append(mask)
    if len(sample_batch) == batch_size:
      yield np.asarray(sample_batch), np.asarray(mask_batch), start_time[i-batch_size:i]
      sample_batch, mask_batch = [], []
    i += 1

  if len(sample_batch) != 0:
    i -= 1  # 出循环时i==segment_num+1
    return np.asarray(sample_batch), np.asarray(mask_batch), start_time[i-batch_size:i]
    # (batch, input_len, mel_bins), (batch, input_len), (batch,)
  

@torch.no_grad()
def inference(
  model: torch.nn.Module, 
  device: torch.device, 
  batch_generator: Generator, 
  detokenize: Callable[[Sequence[int]], np.ndarray]
) -> tuple[float, dict[int, list]]:
  """不同于 train.evaluate
  对 validation, test 集进行评估，返回 pred, target 用于metrics计算
  """

  model.eval()
  begin_time = time.time()
  iteration = 0
  logging.info(f'-------infer starts-------')
  pred_list = []
  # 以 [(start_time, events), ...] 的形式存储曲子对应每个片段的事件tokens

  for sample, mask, start_time in batch_generator:  # (batch, input_len, mel_bins), (batch, input_len), (batch,)
    # Move data to device
    encoder_in, encoder_mask = torch.as_tensor(sample, device=device), torch.as_tensor(mask, device=device)
    predictions = model.generate(inputs_embeds=encoder_in, attention_mask=encoder_mask, do_sample=True)

    iteration += 1
    if iteration % 20 == 0:
      t = time.time() - begin_time
      logging.info(f'infer: {iteration=}, in {t:.3f}s')
      begin_time += t
    # 一般不触发，除非曲子很长
    
    predictions = predictions.cpu().numpy()
    for st, pred in zip(start_time, predictions):
      pred_list.append((st, detokenize(pred)))

  logging.info(f'-------infer exits-------')
  return pred_list


def main(cf: YuiConfig, t5_config: T5Config, audio_path: str, verbose=True) -> note_seq.NoteSequence:
  # Arugments & parameters
  workspace = cf.WORKSPACE
  device = torch.device('cuda') if cf.CUDA and torch.cuda.is_available() else torch.device('cpu')

  # Checkpoint & Log
  checkpoints_dir = os.path.join(workspace, 'checkpoints')
  logs_dir = os.path.join(workspace, 'logs')
  utils.create_logging(logs_dir, f'inference', filemode='w', with_time=True, print_console=verbose)
  resume_checkpoint_path = os.path.join(checkpoints_dir, f'model_resume{cf.MODEL_SUFFIX}.pt')
  best_checkpoint_path = os.path.join(checkpoints_dir, f'model_best{cf.MODEL_SUFFIX}.pt')

  logging.info(cf)
  if device.type == 'cuda':
    logging.info('Using GPU.')
    logging.info(f'GPU number: {torch.cuda.device_count()}')
  else:
    logging.info('Using CPU.')

  # Codec & Vocabulary
  codec = vocabularies.build_codec(cf)  
  vocabulary = vocabularies.Vocabulary(cf, codec.num_classes, extra_ids=cf.EXTRA_IDS)

  # Model
  t5_config = T5Config.from_dict(t5_config)
  model = T5ForConditionalGeneration(config=t5_config)
  logging.info(f'The model has {utils.count_parameters(model):,} trainable parameters')

  # Load statistics & Model
  if os.path.isfile(best_checkpoint_path):
    checkpoint_path = best_checkpoint_path
  elif os.path.isfile(resume_checkpoint_path):
    checkpoint_path = resume_checkpoint_path
  else:
    raise FileNotFoundError(f'{best_checkpoint_path=} or {resume_checkpoint_path=} does not exist')

  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model'])
  model.to(device)

  start_time = time.time()
  detokenize_fn = functools.partial(postprocessors.detokenize, config=cf, vocab=vocabulary)
  batch_generator = audio_split_to_batch(audio_path, cf.BATCH_SIZE, cf_pro_tiny)
  prediction_list = inference(model, device, batch_generator, detokenize_fn)

  ns = postprocessors.event_tokens_to_ns(prediction_list, codec)['ns']
  audio_root, _ = os.path.splitext(audio_path)
  midi_path = f'{audio_root}_yui.midi'
  note_seq.note_sequence_to_midi_file(ns, midi_path)

  logging.info(f'infer finish, time={time.time()-start_time:.3f}s')
  logging.info(f'the output midi file: {midi_path}')
  return ns


if __name__ == '__main__':
  from config.data import YuiConfigDev
  import sys
  import json

  parser = argparse.ArgumentParser(description='命令行中传入一个数字')
  #type是要传入的参数的数据类型  help是该参数的提示信息
  parser.add_argument('--audio', type=str, required=False, help='需要转录的音频文件路径')
  parser.add_argument('--midi', type=str, required=False, help='需要提取钢琴卷帘的MIDI文件路径')
  args = parser.parse_args()
  # sys.stdout.write('#PianoRoll#' + str([[1, 2, 3], [4, -1, 4]]))
  # print(args.midi is None)
  # exit()

  cf_pro_tiny = YuiConfigDev(
    DATASET_DIR=r'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0_hdf5/',
    DATAMETA_NAME=r'maestro-v3.0.0_tinymp3.csv',
    WORKSPACE=r'D:/A日常/大学/毕业设计/code/yui/',
    BATCH_SIZE=8,
    NUM_MEL_BINS=384,
    MODEL_SUFFIX='_kaggle',
  )

  t5_config = config.build_t5_config(
    d_model=cf_pro_tiny.NUM_MEL_BINS,
    vocab_size=669,
    max_length=cf_pro_tiny.MAX_TARGETS_LENGTH,
  )

  audio_path = r'D:/Music/MuseScore/音乐/No,Thank_You.wav'
  # audio_path = r'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0/2017/MIDI-Unprocessed_066_PIANO066_MID--AUDIO-split_07-07-17_Piano-e_3-02_wav--3.wav'
  midi_path = r'D:/Music/MuseScore/乐谱/No,Thank_You.mid'
  # 用的wav/mp3会有影响，而且转录过程具有随机性，目前无法投入实用
  args.midi = args.midi or r'D:/Music/MuseScore/乐谱/Listen!!.mid'

  if args.audio or not args.midi: 
    try:
      audio_path = args.audio or audio_path
      ns = main(cf_pro_tiny, t5_config, audio_path, verbose=args.audio is None)
    except Exception as e:
      logging.exception(e)
      raise e
  else:
    # 未指定audio但指定了midi
    ns = note_seq.midi_file_to_note_sequence(args.midi)
  
  if args.audio or args.midi:
    pianoroll = postprocessors.get_prettymidi_pianoroll(ns)
    sys.stdout.flush()
    data = json.dumps({
      'fps': cf_pro_tiny.PIANOROLL_FPS,
      'pianoroll': postprocessors.get_upr(pianoroll),
      'qpm': ns.tempos[0].qpm,
      # 拿到的是qpm, quarter per minute. PrettyMIDI 拿到的是bpm，跟ns的qpm不同
      # 同时这里忽略了后面可能的tempo变化
    })
    sys.stdout.write(f'##Piano{data}Roll##')
    # flush似乎无效，添加特定标记便于提取内容
