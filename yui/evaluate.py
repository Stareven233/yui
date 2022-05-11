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
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pretty_midi
import note_seq

from datasets import MaestroDataset3, MaestroSamplerEval, collate_fn
import vocabularies
import config
from config.data import YuiConfig
import utils
import postprocessors
import preprocessors


class MetricsViewer:
  def __init__(self, metrics: dict, meta_path: str) -> None:
    self.metrics = metrics
    self.meta_dict = preprocessors.read_metadata(meta_path, split=None)
    self.best_id = np.argmax(self.metrics.get('Onset & offset F1 (0.01) [hist]'))
    logging.info(f"{self.best_id=}, {self.metrics.get('Onset & offset F1 (0.01) [hist]')[self.best_id]}")
    self.sample_num = len(self.metrics['idx_list'])
    self._label_font_dict = {'size': 16}
    self._title_font_dict = {'fontsize': 16}
    self._ticks_size = 16
    # logging.info([self.meta_dict['duration'][i] for i in self.metrics['idx_list']])
  
  def show_pianorolls(self, idx=None, start=0, duration=1000):
    """使用matplotlib绘制钢琴卷帘，shape=(128, frames_len)，不同颜色（值）代表力度"""

    idx = idx or self.best_id
    prs = self.metrics['pianorolls'][idx]
    audio_title = self.meta_dict['canonical_title'][idx]
    # (128, time_len)
    _, axes = plt.subplots(2, 1, figsize=(15, 6), sharex='col')
    end = start + duration
    for i, name in enumerate(('prediction', 'target')):
      axes[i].set_title(F'{name} pianoroll', fontdict=self._title_font_dict)
      axes[i].imshow(prs[i][:, start:end])
      axes[i].set_ylim(0, 128)
      axes[i].set_xticks([0, duration])
      axes[i].set_xticklabels((start, end))
      axes[i].set_yticks([0, 128])
      axes[i].tick_params(labelsize=self._ticks_size)
      axes[i].set_ylabel('pitch', fontdict=self._label_font_dict)
    axes[1].set_xlabel('time (s)', fontdict=self._label_font_dict)
    logging.info(f'generate the pianorolls of {audio_title}')
    plt.show()

  def show_bar_graph(self, metric='F1', bar_num=10):
    obj = f'{metric} [hist]'
    row, col = 3, 2
    _, axes = plt.subplots(row, col, figsize=(6, 16))
    r, c = 0, 0
    x = 0.5 + np.arange(bar_num)
    
    f1_sum = np.zeros((self.sample_num, ))
    for k, v in self.metrics.items():
      if obj not in k:
        continue
      f1_sum += v
    show_ids = np.argsort(f1_sum)[-bar_num:]
    logging.info(f"bar_graph, show: {show_ids=}")
    # 挑最F1最高的bar_num个来展示

    for k, v in self.metrics.items():
      if obj not in k:
        continue

      axes[r, c].set_title(k.removesuffix(' [hist]'), fontdict=self._title_font_dict)
      axes[r, c].bar(x, v[show_ids], color='#f7ba7d', width=0.4, edgecolor='white', linewidth=0.6)
      axes[r, c].set_xticks(x)
      # axes[r, c].set_xticklabels(self.metrics['idx_list'])
      axes[r, c].set_xticklabels(range(bar_num))
      if r == row-1 or r == row-2 and c == col-1:
        axes[r, c].set_xlabel('sample id', fontdict=self._label_font_dict)
      if c == 0:
        axes[r, c].set_ylabel('F1 score', fontdict=self._label_font_dict)
      plt.yticks(size=self._ticks_size)
      plt.xticks(size=self._ticks_size)
      c += 1
      c %= col
      r += int(c == 0)
    
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    if row*col & 1 == 0:
      plt.delaxes(axes[r, c])  
      # 删除最后一个空白的
    plt.show()

  def show_tol_lines(self):
    obj = 'Onset & offset'
    metrics = ('precision', 'recall', 'F1', )
    fmt = ('2--', '3--', 'o-', )
    colors = ('#f8cecc', '#b5f8ff', '#44996c')
    tol = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, )
    plt.figure(figsize=(8, 7))
    for m, f, c in zip(metrics, fmt, colors):
      v = []
      for t in tol:
        key = f'{obj} {m} ({t})'
        v.append(self.metrics[key])
      plt.plot(tol, v, f, markerfacecolor=c, label=f'Onset & Offset {m}')
      plt.yticks(size=14)
      plt.xticks(size=14)
    plt.legend(loc='lower right')
    plt.show()

  def convert_to_midi_file(self, idx=None, path=None):
    idx = idx or self.best_id
    ns = self.metrics['pred_ns_list'][idx]
    abs_id = self.meta_dict['id'][idx]
    if path is None:
      orig_path = self.meta_dict['midi_filename'][idx]
      path = os.path.join('.', os.path.split(orig_path)[1])
      path = '_predicted'.join(os.path.splitext(path))
    path = os.path.abspath(path)
    note_seq.note_sequence_to_midi_file(ns, path)
    logging.info(f'successfully saved the predicted midi file in {path}, with {abs_id=}')

  def show_summary(self):
    skip_keys = {'pred_ns_list', 'pianorolls'}
    for k, v in self.metrics.items():
      if k in skip_keys:
        continue
      if 'events' not in k and 'idx_list' not in k:
        v = np.round(v*100, decimals=2)
      logging.info(f'{k} = {v}')


def show_warmup_curve():
  peak = 1.5e3
  end = 1.5e4
  lr_start = np.arange(1, peak, 1) / 1e6
  lr_end = np.exp(-(6.45 + (np.arange(peak, end, 1) / 3e4)))
  lr = np.hstack((lr_start, lr_end))
  step = np.arange(lr.shape[0])
  plt.plot(step, lr, c='#eb7524', linewidth=1.6)
  plt.xlabel('steps (k)', fontdict={'size': 16})
  plt.ylabel('learning rate', fontdict={'size': 16})
  xticks = np.arange(0, end+10, 5e3)
  xlabels = np.uint16(xticks / 1e3)
  plt.yticks(size=14)
  plt.xticks(ticks=xticks, labels=xlabels, size=14)
  plt.show()


def show_pianoroll(midi_file, fps):
  pm = pretty_midi.PrettyMIDI(midi_file)
  pianorolls = pm.get_piano_roll(fs=fps)
  plt.figure(figsize=(14, 3))
  plt.imshow(pianorolls)
  plt.ylim(0, 128)
  plt.yticks([0, 128])
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

  color_arr = ('#eb7524', '#44996c')
  show_list = ('train_loss', 'eval_loss', )
  # show_list = ('train_loss', )
  plt.figure(figsize=(10, 8))

  for i, k in enumerate(show_list):
    v = statistics[k]
    print(f'{k}={v}')
    print(f'average {k}={sum(v)/len(v)}')
    x = np.arange(len(v))
    ax = plt.subplot(len(show_list), 1, i+1)
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
    if iteration % 5 == 0:
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
  eval_results_path = os.path.join(workspace, 'cache', f'eval_results{cf.MODEL_SUFFIX}.pt')

  logging.info(cf)
  if device.type == 'cuda':
    logging.info('Using GPU.')
    logging.info(f'GPU number: {torch.cuda.device_count()}')
  else:
    logging.info('Using CPU.')

  # Codec & Vocabulary
  codec = vocabularies.build_codec(cf)  
  vocabulary = vocabularies.Vocabulary(cf, codec.num_classes, extra_ids=cf.EXTRA_IDS)

  # Dataset
  meta_path = os.path.join(cf.DATASET_DIR, cf.DATAMETA_NAME)
  dataset = MaestroDataset3(cf.DATASET_DIR, cf, codec, vocabulary, meta_file=cf.DATAMETA_NAME)
  # eval_sampler = MaestroSamplerEval(meta_path, 'validation', batch_size=batch_size, config=cf, sample_num=6)
  eval_sampler = MaestroSamplerEval(meta_path, 'test', batch_size=batch_size, config=cf, sample_num=10)
  eval_loader = DataLoader(dataset=dataset, batch_sampler=eval_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

  if not use_cache:
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

  metrics = postprocessors.calc_metrics(pred_map=pred, target_map=target, codec=codec)
  viewer = MetricsViewer(metrics, meta_path=meta_path)
  viewer.show_summary()
  viewer.show_pianorolls(idx=None, start=100, duration=600)
  viewer.show_bar_graph(bar_num=10)
  viewer.show_tol_lines()
  # viewer.convert_to_midi_file(idx=None)


if __name__ == '__main__':
  from config.data import YuiConfigDev
  cf_pro_tiny = YuiConfigDev(
    # MAX_TARGETS_LENGTH=512,
    DATASET_DIR=r'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0_hdf5/',
    # DATAMETA_NAME=r'maestro-v3.0.0_tiny.csv',
    DATAMETA_NAME=r'maestro-v3.0.0.csv',
    # DATAMETA_NAME=r'maestro-v3.0.0_tinymp3.csv',
    WORKSPACE=r'D:/A日常/大学/毕业设计/code/yui/',

    NUM_WORKERS=2,
    BATCH_SIZE=8,
    NUM_MEL_BINS=384,
    MODEL_SUFFIX='',
    # MODEL_SUFFIX='',
  )

  t5_config = config.build_t5_config(
    d_model=cf_pro_tiny.NUM_MEL_BINS,
    vocab_size=669,
    max_length=cf_pro_tiny.MAX_TARGETS_LENGTH,
  )

  audio = r'D:/Music/MuseScore/音乐/No,Thank_You.wav'
  midi = r'D:/Music/MuseScore/乐谱/No,Thank_You.mid'

  try:
    # main(cf_pro_tiny, t5_config, use_cache=True)
    # main(cf_pro_tiny, t5_config, use_cache=False)
    show_statistics(cf_pro_tiny)
    # show_pianoroll(midi, cf_pro_tiny.PIANOROLL_FPS)
    # show_waveform(audio)
    # show_spectrogram(audio, cf_pro_tiny)
    # show_warmup_curve()
  except Exception as e:
    logging.exception(e)
