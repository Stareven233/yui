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

from datasets import MaestroDataset3, MaestroSampler2, collate_fn
import vocabularies
import config
from config.data import YuiConfig
import utils
import postprocessors
import preprocessors


class MetricsViewer:
  def __init__(self, metrics: dict, meta_dict: dict) -> None:
    self.metrics = metrics
    self.meta_dict = meta_dict
  
  def show_pianorolls(self, idx=0, start=0, duration=1000):
    prs = self.metrics['pianorolls'][idx]
    # (128, time_len)
    _, axes = plt.subplots(2, 1, figsize=(15, 6))
    end = start + duration
    for i, name in enumerate(('prediction', 'target')):
      axes[i].set_title(F'{name} pianoroll')
      axes[i].imshow(prs[i][:, start:end])
      axes[i].set_ylim(0, 128)
      axes[i].set_xticks([0, duration])
      axes[i].set_xticklabels((start, end))
      axes[i].set_yticks([0, 128])
      axes[i].set_xlabel('time (s)')
      axes[i].set_ylabel('pitch ()')
    plt.show()

  def show_p_r_f1(self):
    ...
    # TODO 有时间再补

  def convert_to_midi_file(self, idx=0, path=None):
    ns = self.metrics['pred_ns_list'][idx]
    abs_id = self.meta_dict['id']
    if path is None:
      path = os.path.join('.', os.path.split(self.meta_dict['midi_filename'])[1])
      path = '_predicted'.join(os.path.splitext(path))
    note_seq.note_sequence_to_midi_file(ns, path)
    logging.info(f'successfully saved the predicted midi file in {path}, with {abs_id=}')

  def show_summary(self):
    skip_keys = {'pred_ns_list', 'pianorolls'}
    for k, v in self.metrics.items():
      if k in skip_keys:
        continue
      logging.info(f'{k}={v}')


def show_warmup_curve():
  peak = 1e4
  end = 1e5
  lr_start = np.arange(1, peak, 1) / 1e6
  lr_end = 1. / np.sqrt(np.arange(peak, end, 1))
  lr = np.hstack((lr_start, lr_end))
  step = np.arange(lr.shape[0])
  plt.plot(step, lr, c='#eb7524', linewidth=1.6)
  plt.xlabel('steps (k)', fontdict={'size': 16})
  plt.ylabel('learning rate', fontdict={'size': 16})
  xticks = np.arange(0, end+10, 2e4)
  xlabels = np.uint16(xticks / 1e3)
  plt.yticks(size=14)
  plt.xticks(ticks=xticks, labels=xlabels, size=14)
  plt.show()


def show_pianoroll(midi_file):
  pm = pretty_midi.PrettyMIDI(midi_file)
  pianorolls = pm.get_piano_roll(fs=62.5)
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
  print(statistics)

  color_arr = ('#eb7524', '#44996c')
  show_list = ('train_loss', 'eval_loss', )
  # show_list = ('train_loss', )
  # statistics['train_loss'] = [6.471653413772583, 5.824323949813842, 5.800879411697387, 5.669435682296753, 5.5097521305084225, 5.593990964889526, 5.500483617782593, 5.497598104476928, 5.53036581993103, 5.541115045547485, 5.526965866088867, 5.451460695266723, 5.494182071685791, 5.562448606491089, 5.4498669719696045, 5.491960029602051, 5.322911138534546, 5.533833618164063, 5.392691898345947, 5.546684265136719, 5.420011367797851, 5.44348482131958, 5.4363098192214965, 5.588494310379028, 5.591133661270142, 5.583832712173462, 5.410207343101502, 5.527942070960998, 5.557780809402466, 5.527537384033203, 5.587384743690491, 5.610961380004883, 5.498731417655945, 5.674430031776428, 5.541053700447082, 5.6302377796173095, 5.570145845413208, 5.503081173896789, 5.577783885002137, 5.478591203689575, 5.5206483411788945, 5.513645305633545, 5.5523311805725095, 5.599885425567627, 5.4016321182250975, 5.474492492675782, 5.6176725149154665, 5.4511156749725345, 5.582019348144531, 5.5597440004348755, 5.5489522647857665, 5.510220623016357, 5.595873875617981, 5.538237438201905, 5.568387961387634, 5.514000086784363, 5.538314070701599, 5.5794910717010495, 5.624013199806213, 5.536697630882263, 5.578122553825378, 5.635431275367737, 5.508277044296265, 5.525183424949646, 5.527952375411988, 5.516975293159485, 5.497166800498962, 5.516116995811462, 5.541208171844483, 5.543957600593567, 5.542159523963928, 5.58107394695282, 5.528342571258545, 5.461798758506775, 5.509641213417053, 5.541024966239929, 5.477564244270325, 5.4676784229278566, 5.515361671447754, 5.481831593513489, 5.530974578857422, 5.508228507041931, 5.48603506565094, 5.48190848827362, 5.559149074554443, 5.513868007659912]
  plt.figure(figsize=(10, 8))

  for i, k in enumerate(show_list):
    v = statistics[k]
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

  # Dataset
  meta_path = os.path.join(cf.DATASET_DIR, cf.DATAMETA_NAME)
  dataset = MaestroDataset3(cf.DATASET_DIR, cf, codec, vocabulary, meta_file=cf.DATAMETA_NAME)
  eval_sampler = MaestroSampler2(meta_path, 'train', batch_size=batch_size, config=cf, max_iter_num=-1, drop_last=True)
  # eval_sampler = MaestroSampler2(meta_path, 'test', batch_size=batch_size, config=cf, max_iter_num=-1, drop_last=True)
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

  metrics = postprocessors.calc_full_metrics(pred_map=pred, target_map=target, codec=codec)
  viewer = MetricsViewer(metrics, meta_dict=eval_sampler.meta_dict)
  viewer.show_pianorolls(idx=0, start=200)
  viewer.show_summary()


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

  audio = r'D:/Music/MuseScore/音乐/No,Thank_You.wav'
  midi = r'D:/Music/MuseScore/乐谱/No,Thank_You.mid'

  try:
    # main(cf_pro_tiny, t5_config, use_cache=True)
    main(cf_pro_tiny, t5_config, use_cache=False)
    # show_statistics(cf_pro_tiny)
    # show_pianoroll(midi)
    # show_waveform(audio)
    # show_spectrogram(audio, cf_pro_tiny)
    # show_warmup_curve()
  except Exception as e:
    logging.exception(e)
