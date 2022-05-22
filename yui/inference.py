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
  # audio = utils.z_score(audio)
  audio_len = len(audio)
  segment_len = int(config.segment_second * config.SAMPLE_RATE)
  sample_batch, mask_batch = [], []
  start_pos = np.arange(0, audio_len, segment_len)
  start_time = start_pos / config.SAMPLE_RATE
  start = 0
  segment_num = len(start_pos)
  i = 1
  while i <= segment_num:
  # for i in range(1, segment_num+1):
    start = start_pos[i-1]
    end = audio_len if i==segment_num else start_pos[i]
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
    yield np.asarray(sample_batch), np.asarray(mask_batch), start_time[i-batch_size:i]
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


def main(cf: YuiConfig, t5_config: T5Config, audio_path: str, main=True) -> note_seq.NoteSequence:
  # Arugments & parameters
  workspace = cf.WORKSPACE
  device = torch.device('cuda') if cf.CUDA and torch.cuda.is_available() else torch.device('cpu')

  # Checkpoint & Log
  checkpoints_dir = os.path.join(workspace, 'checkpoints')
  logs_dir = os.path.join(workspace, 'logs')
  utils.create_logging(logs_dir, f'inference', filemode='w', with_time=True, print_console=main)
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
  if main:
    audio_root, _ = os.path.splitext(audio_path)
    midi_path = f'{audio_root}_yui{cf.MODEL_SUFFIX}.midi'
    note_seq.note_sequence_to_midi_file(ns, midi_path)
    logging.info(f'the output midi file: {midi_path}')
    # 作为主程序使用而非在ui里调用

  logging.info(f'infer finish, time={time.time()-start_time:.3f}s')
  return ns


if __name__ == '__main__':
  from config.data import YuiConfigDev
  import sys
  import json

  parser = argparse.ArgumentParser(description='命令行中传入一个数字')
  #type是要传入的参数的数据类型  help是该参数的提示信息
  parser.add_argument('--audio', type=str, required=False, help='需要转录的音频文件路径')
  parser.add_argument('--midi', type=str, required=False, help='需要提取钢琴卷帘的MIDI文件路径')
  parser.add_argument('--upr', type=str, required=False, help='需要导出为MIDI的upr json')
  args = parser.parse_args()
  # sys.stdout.write('#PianoRoll#' + str([[1, 2, 3], [4, -1, 4]]))
  # print(args.midi is None)
  # exit()

  cf_pro_tiny = YuiConfigDev(
    DATASET_DIR=r'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0_hdf5/',
    DATAMETA_NAME=r'maestro-v3.0.0_tinymp3.csv',
    WORKSPACE=r'F:/CODE/Pythoncode/LittleWorks/yui/',
    BATCH_SIZE=8,
    NUM_MEL_BINS=384,
    # MODEL_SUFFIX='_k20.1',
    # MODEL_SUFFIX='_k22.1',
    # MODEL_SUFFIX='_k23',
    MODEL_SUFFIX='_k24',
    # MODEL_SUFFIX='_k25',
    # MODEL_SUFFIX='',
  )

  t5_config = config.build_t5_config(
    d_model=cf_pro_tiny.NUM_MEL_BINS,
    vocab_size=669,
    max_length=cf_pro_tiny.MAX_TARGETS_LENGTH,
  )

  audio_path = r'D:/Music/MuseScore/乐谱/No,Thank_You.wav'
  audio_path = r'D:/Music/MuseScore/乐谱/卡农.wav'
  audio_path = r'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0/2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--3.wav'
  audio_path = r'D:/Music/MuseScore/乐谱/欢乐颂.wav'
  # 用的wav/mp3会有影响，而且转录过程具有随机性，目前无法投入实用
  # midi_path = r'D:/Music/MuseScore/乐谱/欢乐颂.mid'

  if args.upr:  # args.upr: 导出的midi路径
  # if True:
    # f = open(r'D:\Music\MuseScore\乐谱\a.upr', 'r')
    # uprJSON = f.read()
    # f.close()
    uprJSON = sys.stdin.read()
    upr = json.loads(uprJSON)
    pianoroll = postprocessors.upr_to_pianoroll(upr['pianoroll'])
    ts = pretty_midi.containers.TimeSignature(*upr['timeSignature'], time=0)
    ks = pretty_midi.containers.KeySignature(upr['keySignature'], time=0)
    bpm = pretty_midi.qpm_to_bpm(upr['qpm'], ts.numerator, ts.denominator)
    pm = postprocessors.piano_roll_to_pretty_midi(pianoroll, fs=upr['fps'], tempo=bpm)
    pm.time_signature_changes.append(ts)
    pm.key_signature_changes.append(ks)
    pm.write(args.upr)
    # pm.write(r'D:/Music/MuseScore/乐谱/No,Thank_You_key.mid')
    # exit()
  elif audio_path := args.audio or audio_path: 
    try:
      ns = main(cf_pro_tiny, t5_config, audio_path, main=args.audio is None)
    except Exception as e:
      logging.exception(e)
      raise e
  elif args.midi:
    # 未指定audio但指定了midi
    # ns = note_seq.midi_file_to_note_sequence(args.midi)
    pm = utils.UnsoundPM(args.midi)
    ns = note_seq.midi_to_note_sequence(pm)
  
  if args.audio or args.midi:
  # if True:
    pianoroll = postprocessors.get_prettymidi_pianoroll(ns)
    ts, ks = ns.time_signatures, ns.key_signatures
    ts = len(ts) > 0 and ts[0] or utils.default_ts
    ks = len(ks) > 0 and ks[0] or utils.default_ks
    # 拍号调号在乐曲中途可能会变换，但这里只取第一个（出现时间设为0）
    qpm = ns.tempos[0].qpm if bool(ns.tempos) else 120

    sys.stdout.flush()
    data = json.dumps({
      'fps': cf_pro_tiny.PIANOROLL_FPS,
      'pianoroll': postprocessors.pianoroll_to_upr(pianoroll),
      'qpm': qpm,
      # 拿到的是qpm, quarter per minute. PrettyMIDI 拿到的是bpm，跟ns的qpm不同
      # 同时这里忽略了后面可能的tempo变化
      'timeSignature': [ts.numerator, ts.denominator],
      # 拍号应该也是估计的，未必准确
      'keySignature': ks.key + ks.mode*12,
      # mode=0: major, =1: minor; 转换成pretty_midi的key_number
    })
    sys.stdout.write(f'##Piano{data}Roll##')
    # flush似乎无效，添加特定标记便于提取内容
