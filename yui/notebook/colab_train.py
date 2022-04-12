import os
import time
import logging
import sys
sys.path.insert(0, r'/content/yui')

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Config
from transformers.optimization import Adafactor, AdafactorSchedule

from datasets import MaestroDataset, MaestroSampler2, collate_fn
import vocabularies
import config
from config.data import YuiConfigPro
import utils
from train import train, evaluate

# config
cf = YuiConfigPro(
  DATASET_DIR=r'/content/maestro/',
  DATAMETA_NAME=r'maestro-v3.0.0.csv',
  WORKSPACE=r'/content/',
  CUDA=False,
  NUM_EPOCHS=2,
  BATCH_SIZE=8,
  TRAIN_ITERATION=100,
)
# Codec & Vocabulary
codec = vocabularies.build_codec(cf)
vocabulary = vocabularies.Vocabulary(cf, codec.num_classes, extra_ids=cf.EXTRA_IDS)
t5_config = config.build_t5_config(
  vocab_size=vocabulary.vocab_size
)

# Arugments & parameters
workspace = cf.WORKSPACE
batch_size = cf.BATCH_SIZE
device = torch.device('cuda') if cf.CUDA and torch.cuda.is_available() else torch.device('cpu')
num_workers = cf.NUM_WORKERS

# Checkpoint & Log
checkpoints_dir = os.path.join(workspace, 'checkpoints')
utils.create_folder(checkpoints_dir)
logs_dir = os.path.join(workspace, 'logs')
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

resume = True

# Dataset
meta_path = os.path.join(cf.DATASET_DIR, cf.DATAMETA_NAME)

train_sampler = MaestroSampler2(meta_path, 'train', batch_size=batch_size, config=cf, max_iter_num=cf.TRAIN_ITERATION)
train_dataset = MaestroDataset(cf.DATASET_DIR, cf, codec, vocabulary, meta_file=cf.DATAMETA_NAME)
train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=False)

validate_sampler = MaestroSampler2(meta_path, 'validation', batch_size=batch_size, config=cf, max_iter_num=-1)
validate_loader = DataLoader(dataset=train_dataset, batch_sampler=validate_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=False)
# pin_memory: 锁页内存，不会与虚存进行交换，转到gpu时快一些，但很容易超出gpu显存

# Model
t5_config = T5Config.from_dict(t5_config)
model = T5ForConditionalGeneration(config=t5_config)
logging.info(f'The model has {model.num_parameters():,} trainable parameters')
# 17,896 for dev; 48,626,048 for pro; while T5-Small has 60 million parameters

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

if not resume:
  ...
  # 从头开始训练模型
elif not os.path.isfile(resume_checkpoint_path):
  logging.info(f'{resume_checkpoint_path=} does not exist, train from scratch')
elif not os.path.isfile(statistics_path):
  logging.info(f'{statistics_path=} does not exist, train from scratch')
else:
  statistics = torch.load(statistics_path)
  # 单独保存后面数据分析读取方便些
  # raise FileNotFoundError(f'{resume_checkpoint_path=} does not exist')
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
  logging.info(f'resume training with epoch={resume_epoch}, lr={learning_rate}')

# Loss function
criterion = torch.nn.CrossEntropyLoss(ignore_index=cf.PAD_ID)

# Optimizer
optimizer = Adafactor(model.parameters(), lr=learning_rate, scale_parameter=False, relative_step=False, warmup_init=False)
scheduler = AdafactorSchedule(optimizer, learning_rate)

model.to(device)
epoch = resume_epoch
loop_start_time = time.time()
start_time = time.time()
assert epoch == train_sampler.epoch, f'resume training: {epoch=} != {train_sampler.epoch=}'
logging.info(f'-------train loop starts, {start_time=:.3f}s-------')

# for epoch in range(resume_epoch, cf.NUM_EPOCHS):
while epoch < cf.NUM_EPOCHS:
  train_loss = train(model, device, train_loader, criterion, optimizer, scheduler)
  statistics['train_loss'].append(train_loss)
  current_lr = scheduler.get_lr()

  # 训练数据完整采样一轮
  if train_sampler.epoch > epoch:
    validate_sampler.reset_state()
    validate_loss = evaluate(model, device, validate_loader, criterion)
    statistics['eval_loss'].append(train_loss)
    # 等train数据完整过了一遍再进行评估
    logging.info(
      f'{epoch=} finish, time={time.time()-start_time:.3f}s, {train_loss=}, {validate_loss=}'
      f', with lr={current_lr}'
    )

    early_stopping(validate_loss)
    if early_stopping.stop:
      logging.info(f'early stoping')
      break

    epoch += 1
    start_time = time.time()
    train_sampler.reset_state()
  
  # Save model
  statistics['epoch'] = epoch
  checkpoint = {
    'epoch': epoch,
    'model': model.state_dict(),
    'sampler': train_sampler.state_dict(),
    'learning_rate': current_lr,
    'early_stopping': early_stopping.state_dict(),
  }
  torch.save(checkpoint, resume_checkpoint_path)
  torch.save(statistics, statistics_path)
  logging.info(f'save model and statistics to {checkpoints_dir}')
logging.info(f'-------train loop ends, time={time.time()-loop_start_time:.3f}s-------')
