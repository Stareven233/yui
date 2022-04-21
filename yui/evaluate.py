"""evaluate
用于各项 metrics 计算、图表绘制及展示
"""

import os
import time
import logging
from collections import defaultdict
import functools
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Config
import librosa
import matplotlib.pyplot as plt
import librosa.display
import pretty_midi

from datasets import MaestroDataset3, MaestroSampler2, collate_fn
import vocabularies
import config
from config.data import YuiConfig
import utils
import postprocessors
import preprocessors


def show_pianorolls(midi_file):
  pm = pretty_midi.PrettyMIDI(midi_file)
  pianorolls = pm.get_piano_roll()
  plt.figure(figsize=(12, 3))
  plt.imshow(pianorolls)
  plt.show()


def show_waveform(audio_file):
  x, sr=librosa.load(audio_file)
  plt.figure(figsize=(16, 5))
  librosa.display.waveplot(x, sr=sr)
  plt.show()


def show_spectrogram(audio_file, config=YuiConfig):
  x, _=librosa.load(audio_file)
  x = preprocessors.compute_spectrograms({'inputs': x}, config)
  spectrogram = x['inputs']
  plt.figure(figsize=(3, 6))
  plt.imshow(spectrogram)
  plt.show()


def show_statistics(cf: YuiConfig):
  path = os.path.join(cf.WORKSPACE, 'checkpoints', f'statistics{cf.MODEL_SUFFIX}.pt')
  statistics = torch.load(path)
  print(statistics)
  plt.figure(figsize=(5, 8))

  color_arr = ('#e87e20', '#166fcf')
  for i, k in enumerate(('train_loss', 'eval_loss', )):
    v = statistics[k]
    print(f'average {k}={sum(v)/len(v)}')
    x = np.arange(len(v))
    ax = plt.subplot(2, 1, i+1)
    ax.set_title(k)
    ax.plot(x, v, c=color_arr[i], linewidth=1.6)
    ax.grid(True)  # 显示网格线

  plt.show()


@torch.no_grad()
def evaluate(
  model: torch.nn.Module, 
  device: torch.device, 
  data_loader: DataLoader, 
  detokenize: Callable[[Sequence[int]], np.ndarray]
) -> tuple[float, dict[int, list]]:
  """不同于 train.evaluate
  对 validation, test 集进行评估，返回 pred, target 用于metrics计算
  """

  model.eval()
  begin_time = time.time()
  iteration = 0
  logging.info(f'-------eval starts-------')
  pred_map = defaultdict(list)
  target_map = defaultdict(list)
  # 以 {audio_id: [(start_time, events), ...]} 的形式存储曲子对应每个片段的事件tokens
  # 其中events的shape=(n, ); target是处理完的事件而非原本的note_sequences，或许有差异，但原本就是基于target学习的，metrics计算自然也要以其为准

  for batch_data_dict in data_loader:
    # Move data to device
    encoder_in, _, _, target, _ = utils.move_to_device(batch_data_dict, device)
    pred = model.generate(inputs_embeds=encoder_in, do_sample=True)
    # generate 返回的就是一个token序列，跟target对应；用这个不能算交叉熵，可换成其他的标准
    # logging.info(get_feature_desc(out))

    iteration += 1
    if iteration % 20 == 0:
      t = time.time() - begin_time
      logging.info(f'eval: {iteration=}, in {t:.3f}s')
      begin_time += t
    
    pred = pred.cpu().numpy()
    target = batch_data_dict['decoder_target_tokens']  # type==np.ndarray
    for i, meta in enumerate(batch_data_dict['id']):
      idx, st = eval(meta)  # (2, 5.801)
      pred_map[idx].append((st, detokenize(pred[i])))
      target_map[idx].append((st, detokenize(target[i])))
      # 先detokenize，与之前的tokenize对应，才是真正的模型预测输出，用于后面metrics及换回ns

  logging.info(f'-------eval exits-------')
  return pred_map, target_map


def main(cf: YuiConfig, t5_config: T5Config, use_cache: bool=False):
  # Arugments & parameters
  workspace = cf.WORKSPACE
  batch_size = cf.BATCH_SIZE
  device = torch.device('cuda') if cf.CUDA and torch.cuda.is_available() else torch.device('cpu')
  num_workers = cf.NUM_WORKERS

  # Checkpoint & Log
  checkpoints_dir = os.path.join(workspace, 'checkpoints')
  utils.create_folder(checkpoints_dir)
  logs_dir = os.path.join(workspace, 'logs')
  utils.create_logging(logs_dir, f'eval', filemode='w', with_time=True)
  resume_checkpoint_path = os.path.join(checkpoints_dir, f'model_resume{cf.MODEL_SUFFIX}.pt')
  best_checkpoint_path = os.path.join(checkpoints_dir, f'model_best{cf.MODEL_SUFFIX}.pt')
  statistics_path = os.path.join(checkpoints_dir, f'statistics{cf.MODEL_SUFFIX}.pt')
  eval_results_path = os.path.join(checkpoints_dir, f'eval_results{cf.MODEL_SUFFIX}.pt')

  logging.info(cf)
  if device.type == 'cuda':
    logging.info('Using GPU.')
    logging.info(f'GPU number: {torch.cuda.device_count()}')
  else:
    logging.info('Using CPU.')

  # Codec & Vocabulary
  codec = vocabularies.build_codec(cf)  
  vocabulary = vocabularies.Vocabulary(cf, codec.num_classes, extra_ids=cf.EXTRA_IDS)

  if not use_cache:
    # Dataset
    meta_path = os.path.join(cf.DATASET_DIR, cf.DATAMETA_NAME)
    dataset = MaestroDataset3(cf.DATASET_DIR, cf, codec, vocabulary, meta_file=cf.DATAMETA_NAME)
    eval_sampler = MaestroSampler2(meta_path, 'train', batch_size=batch_size, config=cf, max_iter_num=-1, drop_last=True)
    # eval_sampler = MaestroSampler2(meta_path, 'test', batch_size=batch_size, config=cf, max_iter_num=-1, drop_last=True)
    eval_loader = DataLoader(dataset=dataset, batch_sampler=eval_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    # Model
    t5_config = T5Config.from_dict(t5_config)
    model = T5ForConditionalGeneration(config=t5_config)
    logging.info(f'The model has {utils.count_parameters(model):,} trainable parameters')  # 15,843,712

    # Load statistics & Model
    if not os.path.isfile(statistics_path):
      raise FileNotFoundError(f'{statistics_path=} does not exist')

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
    pred, target = evaluate(model, device, eval_loader, detokenize_fn)
    torch.save((pred, target, ), eval_results_path)
    logging.info(f'eval finish, time={time.time()-start_time:.3f}s')

  else:
    if not os.path.isfile(eval_results_path):
      raise FileNotFoundError(f'{eval_results_path=} does not exist')
    pred, target = torch.load(eval_results_path)

  metrics = postprocessors.calc_full_metrics(pred_map=pred, target_map=target, codec=codec)
  logging.info(f'{metrics=}')


if __name__ == '__main__':
  from config.data import YuiConfigDev
  cf_pro_tiny = YuiConfigDev(
    # MAX_TARGETS_LENGTH=512,
    DATASET_DIR=r'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0_hdf5/',
    # DATASET_DIR=r'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0/',
    # DATAMETA_NAME=r'maestro-v3.0.0_tiny.csv',
    # DATAMETA_NAME=r'maestro-v3.0.0.csv',
    DATAMETA_NAME=r'maestro-v3.0.0_tinymp3.csv',
    WORKSPACE=r'D:/A日常/大学/毕业设计/code/yui/',

    BATCH_SIZE=8,
    NUM_MEL_BINS=256,
    MODEL_SUFFIX='_kaggle',
  )
  # 用于本地测试的pro配置

  t5_config = config.build_t5_config(
    # d_kv=32,
    # d_ff=256,
    # num_layers=2,
    # num_decoder_layers=2,
    # num_heads=4,
    d_model=cf_pro_tiny.NUM_MEL_BINS,
    vocab_size=769,
    max_length=cf_pro_tiny.MAX_TARGETS_LENGTH,
  )

  try:
    # main(cf_pro_tiny, t5_config, use_cache=True)
    # main(cf_pro_tiny, t5_config, use_cache=False)
    show_statistics(cf_pro_tiny)
  except Exception as e:
    logging.exception(e)

  # audio = r'D:/Music/MuseScore/音乐/No,Thank_You.wav'
  # midi = r'D:/Music/MuseScore/乐谱/No,Thank_You.mid'
  # # show_waveform(audio)
  # # show_spectrogram(audio, cf_pro_tiny)
  # show_pianorolls(midi)
