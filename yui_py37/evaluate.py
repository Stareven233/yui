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

from datasets import MaestroDataset, MaestroSampler2, collate_fn
import vocabularies
import config
from config.data import YuiConfig
import utils
import postprocessors


@torch.no_grad()
def evaluate(
  model, 
  device, 
  data_loader, 
  criterion,
  detokenize
):
  """不同于 train.evaluate
  对 validation, test 集进行评估，返回 pred, target 用于metrics计算
  """

  model.eval()
  begin_time = time.time()
  iteration = 0
  epoch_loss = 0
  logging.info(f'-------eval starts-------')
  pred_map = defaultdict(list)
  target_map = defaultdict(list)
  # 以 {audio_id: [(start_time, events), ...]} 的形式存储曲子对应每个片段的事件tokens
  # 其中events的shape=(n, ); target是处理完的事件而非原本的note_sequences，或许有差异，但原本就是基于target学习的，metrics计算自然也要以其为准

  for batch_data_dict in data_loader:
    # Move data to device
    encoder_in, encoder_mask, decoder_in, target, target_mask = utils.move_to_device(batch_data_dict, device)
    
    out = model(
      inputs_embeds=encoder_in, 
      attention_mask=encoder_mask, 
      decoder_input_ids=decoder_in, 
      decoder_attention_mask=target_mask
    )
    # logging.info(get_feature_desc(out))

    logits = out.logits
    # logits: (2, 1024, 6000)[batch, target_len, classes] -> (2048, 6000); target: (2, 1024) -> (2048)
    loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
    loss = loss.item()
    epoch_loss += loss

    iteration += 1
    if iteration % 20 == 0:
      t = time.time() - begin_time
      logging.info(f'eval: iteration={iteration}, loss={loss}, in {t:.3f}s')
      begin_time += t
    
    pred = np.argmax(logits.cpu().numpy(), axis=-1)
    target = batch_data_dict['decoder_target_tokens']  # type==np.ndarray
    for i, meta in enumerate(batch_data_dict['id']):
      idx, st = eval(meta)  # (2, 5.801)
      pred_map[idx].append((st, detokenize(pred[i])))
      target_map[idx].append((st, detokenize(target[i])))
      # 先detokenize，与之前的tokenize对应，才是真正的模型预测输出，用于后面metrics及换回ns

  logging.info(f'-------eval exits-------')
  avg_loss = epoch_loss / iteration
  return avg_loss, pred_map, target_map


def main(cf, use_cache=False):
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
  resume_checkpoint_path = os.path.join(checkpoints_dir, 'model_resume.pt')
  best_checkpoint_path = os.path.join(checkpoints_dir, 'model_best.pt')
  statistics_path = os.path.join(checkpoints_dir, 'statistics.pt')
  eval_results_path = os.path.join(checkpoints_dir, 'eval_results.pt')

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
    dataset = MaestroDataset(cf.DATASET_DIR, cf, codec, vocabulary, meta_file=cf.DATAMETA_NAME)
    eval_sampler = MaestroSampler2(meta_path, 'test', batch_size=batch_size, config=cf, max_iter_num=-1, drop_last=True)
    eval_loader = DataLoader(dataset=dataset, batch_sampler=eval_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    # Model
    t5_config = config.build_t5_config(vocab_size=vocabulary.vocab_size)
    t5_config = T5Config.from_dict(t5_config)
    model = T5ForConditionalGeneration(config=t5_config)
    logging.info(f'The model has {utils.count_parameters(model):,} trainable parameters')  # 15,843,712

    # Load statistics & Model
    if not os.path.isfile(statistics_path):
      raise FileNotFoundError(f'{statistics_path=} does not exist')
    statistics = torch.load(statistics_path)
    logging.info(statistics)

    if os.path.isfile(best_checkpoint_path):
      checkpoint_path = best_checkpoint_path
    elif os.path.isfile(resume_checkpoint_path):
      checkpoint_path = resume_checkpoint_path
    else:
      raise FileNotFoundError(f'{best_checkpoint_path=} or {resume_checkpoint_path=} does not exist')

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    # Loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=cf.PAD_ID)
    # ignore_index 指定的class不会参与计算，target/pred都是 -> pad部分在pred应当会是随机数?反正不重要

    model.to(device)
    start_time = time.time()
    detokenize_fn = functools.partial(postprocessors.detokenize, config=cf, vocab=vocabulary)
    eval_loss, pred, target = evaluate(model, device, eval_loader, criterion, detokenize_fn)
    torch.save((eval_loss, pred, target, ), eval_results_path)
    logging.info(f'eval finish, time={time.time()-start_time:.3f}s, {eval_loss=}')

  else:
    if not os.path.isfile(eval_results_path):
      raise FileNotFoundError(f'{eval_results_path=} does not exist')
    eval_loss, pred, target = torch.load(eval_results_path)

  metrics = postprocessors.calc_full_metrics(pred_map=pred, target_map=target, codec=codec)
  logging.info(f'metrics={metrics}')


if __name__ == '__main__':
  from config.data import YuiConfigDev
  cf = YuiConfigDev(
    CUDA=True
  )
  try:
    main(cf, use_cache=True)
    # main(cf, use_cache=False)
  except Exception as e:
    logging.exception(e)
