import os
import time
import logging

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Config
from transformers.optimization import Adafactor, AdafactorSchedule

from datasets import MaestroDataset2, MaestroSampler2, collate_fn
import vocabularies
import config
from config.data import YuiConfig
import utils


def train(
  model: torch.nn.Module, 
  device: torch.device, 
  data_loader: DataLoader, 
  criterion: torch.nn.Module, 
  optimizer: torch.optim.Optimizer,
  scheduler: torch.optim.lr_scheduler.LambdaLR,
  accumulation_steps: int,
) -> float:

  model.train()
  begin_time = time.time()
  iteration = 0
  epoch_avg_loss = 0
  epoch = data_loader._index_sampler.epoch
  logging.info(f'-------train starts, epoch={epoch}-------')
  d_time = time.time()

  for batch_data_dict in data_loader:
    # Move data to device    
    logging.debug(f'-------train, iteration={iteration}-------')
    encoder_in, encoder_mask, decoder_in, target, target_mask = utils.move_to_device(batch_data_dict, device)
    logging.debug(f'data: {time.time() - d_time:.3f}s')
    logging.debug(f'data_ids: {[eval(t)[0] for t in batch_data_dict["id"]]}')
    m_time = time.time()

    out = model(
      inputs_embeds=encoder_in, 
      attention_mask=encoder_mask, 
      decoder_input_ids=decoder_in, 
      decoder_attention_mask=target_mask
    )

    logits = out.logits
    del out
    # sequence_output.shape=torch.Size([2, 1024, 512]) -> lm_logits.shape=torch.Size([2, 1024, 6000])
    # 两个batch，每个由512词向量表达，通过仿射层变为6000个类别
    loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1)) / accumulation_steps
    # logits: (2, 1024, 6000)[batch, target_len, classes] -> (2048, 6000); target: (2, 1024) -> (2048)

    # Backward
    loss.backward()
    if (iteration + 1) % accumulation_steps == 0:
      optimizer.step()
      optimizer.zero_grad()
      # 梯度累加 gradient accumulation

    loss = loss.item()
    iteration += 1
    epoch_avg_loss = (epoch_avg_loss*(iteration - 1) + loss) / iteration
    if iteration % 50 == 0:
      t = time.time() - begin_time
      logging.info(f'train: epoch={epoch}, iteration={iteration}, loss={loss}, lr={scheduler.get_lr()}, in {t:.3f}s')
      # logging.info(f'id={batch_data_dict["id"].tolist()}')
      begin_time += t

    logging.debug(f'model: {time.time() - m_time:.3f}s')
    d_time = time.time()
    
  logging.info(f'-------train exits, epoch={epoch}-------')
  return epoch_avg_loss


@torch.no_grad()
def evaluate(
  model: torch.nn.Module, 
  device: torch.device, 
  data_loader: DataLoader, 
  criterion: torch.nn.Module
) -> float:
  # 损失函数除了作为模型训练时候的优化目标，也能够作为模型好坏的一种评价指标。但通常人们还会从其它角度评估模型的好坏，这就是评估指标。
  # 通常损失函数都可以作为评估指标，如MAE,MSE,CategoricalCrossentropy等也是常用的评估指标。
  # 但评估指标不一定可以作为损失函数，例如AUC,Accuracy,Precision。因为评估指标不要求连续可导，而损失函数通常要求连续可导。

  model.eval()
  begin_time = time.time()
  iteration = 0
  epoch_loss = 0
  epoch = data_loader._index_sampler.epoch
  logging.info(f'-------eval starts, epoch={epoch}-------')

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
    loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
    loss = loss.item()
    epoch_loss += loss

    iteration += 1
    if iteration % 20 == 0:
      t = time.time() - begin_time
      logging.info(f'eval: epoch={epoch}, iteration={iteration}, loss={loss}, in {t:.3f}s')
      begin_time += t

  logging.info(f'-------eval exits, epoch={epoch}-------')
  return epoch_loss / iteration


def main(cf: YuiConfig, t5_config: T5Config, resume: bool=False):
  # Arugments & parameters
  batch_size = cf.BATCH_SIZE
  device = torch.device('cuda') if cf.CUDA and torch.cuda.is_available() else torch.device('cpu')
  num_workers = cf.NUM_WORKERS

  # Checkpoint & Log
  checkpoints_dir = os.path.join(cf.WORKSPACE, 'checkpoints')
  utils.create_folder(checkpoints_dir)
  logs_dir = os.path.join(cf.WORKSPACE, 'logs')
  utils.create_logging(logs_dir, f'train', filemode='w', with_time=True)
  resume_checkpoint_path = os.path.join(checkpoints_dir, 'model_resume.pt')
  best_checkpoint_path = os.path.join(checkpoints_dir, 'model_best.pt')
  statistics_path = os.path.join(checkpoints_dir, 'statistics.pt')

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

  train_sampler = MaestroSampler2(meta_path, 'train', batch_size=batch_size, config=cf, max_iter_num=cf.TRAIN_ITERATION)
  train_dataset = MaestroDataset2(cf.DATASET_DIR, cf, codec, vocabulary, meta_file=cf.DATAMETA_NAME)
  train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=False)
  
  validate_sampler = MaestroSampler2(meta_path, 'validation', batch_size=batch_size, config=cf, max_iter_num=-1)
  validate_loader = DataLoader(dataset=train_dataset, batch_sampler=validate_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=False)
  # pin_memory: 锁页内存，不会与虚存进行交换，转到gpu时快一些，但很容易超出gpu显存
  # dataset一致，主要是抽取方式sampler不同

  # Model
  t5_config = T5Config.from_dict(t5_config)
  model = T5ForConditionalGeneration(config=t5_config)
  logging.info(f'The model has {model.num_parameters():,} trainable parameters')
  # 17,896 for dev; 48,626,048 for pro; while T5-Small has 60 million parameters
  model.to(device)

  # Early stop
  early_stopping = utils.EarlyStopping(
    best_path=best_checkpoint_path,
    resume_path=resume_checkpoint_path,
    patience=cf.OVERFIT_PATIENCE, 
    verbose=True
  )

  # Resume training
  resume_epoch = 0
  learning_rate = cf.LEARNING_RATE
  statistics = {
    'epoch': 0,
    'train_loss': [],
    'eval_loss': []
  }

  # Loss function
  criterion = torch.nn.CrossEntropyLoss(ignore_index=cf.PAD_ID)
  # Optimizer
  # optimizer = Adafactor(model.parameters(), lr=learning_rate, scale_parameter=False, relative_step=False, warmup_init=False)
  optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
  scheduler = AdafactorSchedule(optimizer, learning_rate)

  if not resume:
    ...
    # 从头开始训练模型
  elif not os.path.isfile(resume_checkpoint_path):
    logging.info(f'resume_checkpoint_path={resume_checkpoint_path} does not exist, train from scratch')
  elif not os.path.isfile(statistics_path):
    logging.info(f'statistics_path={statistics_path} does not exist, train from scratch')
  else:
    statistics = torch.load(statistics_path)
    # 单独保存后面数据分析读取方便些
    # raise FileNotFoundError(f'resume_checkpoint_path={resume_checkpoint_path} does not exist')
    checkpoint = torch.load(resume_checkpoint_path)
    # 以TRAIN_ITERATION为单位保存checkpoint
    early_stopping.load_state_dict(checkpoint['early_stopping'])

    model.load_state_dict(checkpoint['model'])
    train_sampler.load_state_dict(checkpoint['sampler'])
    validate_sampler.epoch = train_sampler.epoch
    # 二者epoch一致
    resume_epoch = checkpoint['epoch']
    learning_rate = checkpoint['learning_rate'][-1]
    # scheduler.get_lr 拿到的lr是个列表
    optimizer.load_state_dict(checkpoint['optimizer'])
    logging.info(f'resume training with epoch={resume_epoch}, lr={learning_rate}')

  epoch = resume_epoch
  loop_start_time = time.time()
  start_time = time.time()
  assert epoch == train_sampler.epoch, f"resume training: epoch={epoch} != train_sampler.epoch={train_sampler.epoch}"
  logging.info(f'-------train loop starts, start_time={start_time:.3f}s-------')

  # for epoch in range(resume_epoch, cf.NUM_EPOCHS):
  while epoch < cf.NUM_EPOCHS:
    optimizer.zero_grad()
    train_loss = train(model, device, train_loader, criterion, optimizer, scheduler, accumulation_steps=cf.accumulation_steps)
    statistics['train_loss'].append(train_loss)
    current_lr = scheduler.get_lr()

    # 训练数据完整采样一轮
    if train_sampler.epoch > epoch:
      validate_sampler.reset_state()
      validate_loss = evaluate(model, device, validate_loader, criterion)
      statistics['eval_loss'].append(train_loss)
      # 等train数据完整过了一遍再进行评估
      logging.info(
        f'epoch={epoch} finish, time={time.time()-start_time:.3f}s, train_loss={train_loss}, validate_loss={validate_loss}'
        f', with lr={current_lr}'
      )

      early_stopping(validate_loss)
      if early_stopping.stop:
        logging.info(f'early stoping')
        break

      epoch += 1
      start_time = time.time()
      train_sampler.reset_state()
      # 重新设置sampler状态，使下一epoch切片方式有所不同
    
    # Save model
    statistics['epoch'] = epoch
    checkpoint = {
      'epoch': epoch,
      'model': model.state_dict(),
      'sampler': train_sampler.state_dict(),
      'learning_rate': current_lr,
      'early_stopping': early_stopping.state_dict(),
      'optimizer': optimizer.state_dict(),
    }
    # 上面的evaluate对train_sampler的影响也要记录下来
    # epoch要等上面计算完才能保存，保证二者记录的epoch一致
    torch.save(checkpoint, resume_checkpoint_path)
    torch.save(statistics, statistics_path)
    logging.info(f'save model and statistics to {checkpoints_dir}')
  logging.info(f'-------train loop ends, time={time.time()-loop_start_time:.3f}s-------')


if __name__ == '__main__':
  from config.data import YuiConfigPro, YuiConfigDev

  cf_pro_tiny = YuiConfigPro(
    BATCH_SIZE=4,
    NUM_WORKERS=3,
    NUM_EPOCHS=160,
    # MAX_TARGETS_LENGTH=300,
    DATASET_DIR=r'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0/',
    # DATAMETA_NAME=r'maestro-v3.0.0_tiny.csv',
    DATAMETA_NAME=r'maestro-v3.0.0_tinymp3.csv',
    WORKSPACE=r'D:/A日常/大学/毕业设计/code/yui/',
    TRAIN_ITERATION=600,
    LEARNING_RATE=1e-3,

    NUM_MEL_BINS=256,
    STEPS_PER_SECOND=100,  # 降低精度到1ms看看能不能收敛
  )
  cf_dev_tiny = YuiConfigDev(
    BATCH_SIZE=4,  # 16 will: CUDA out of memory (4GB)
  )
  # 用于本地测试的pro配置

  t5_config = config.build_t5_config(
    vocab_size=4449,
    num_layers=2,
    num_decoder_layers=2, 
    d_model=256,
    d_kv=64,
    d_ff=256,
    num_layers=2,
    num_decoder_layers=2,
    num_heads=4,
    dropout_rate=0.1,
    num_beams=4,
    vocab_size=4449,
  )

  try:
    # main(cf_pro_tiny, resume=True)
    main(cf_pro_tiny, t5_config, resume=False)
  except Exception as e:
    logging.exception(e)
