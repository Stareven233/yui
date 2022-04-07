import os
from pickle import TRUE
import time
import logging
import shutil

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Config
from transformers.optimization import Adafactor, AdafactorSchedule

from datasets import MaestroDataset, MaestroSampler2, collate_fn
import vocabularies
import config
from config.data import YuiConfig
import utils
from utils import get_feature_desc


def train(
  model: torch.nn.Module, 
  device: torch.device, 
  data_loader: DataLoader, 
  criterion: torch.nn.Module, 
  optimizer: torch.optim.Optimizer,
  scheduler: torch.optim.lr_scheduler.LambdaLR,
  statistics: dict
) -> float:

  model.train()
  begin_time = time.time()
  iteration = 0
  epoch_loss = 0
  epoch = data_loader._index_sampler.epoch
  logging.info(f'-------train starts, {epoch=}-------')

  for batch_data_dict in data_loader:
    # logging.info(f'{get_feature_desc(batch_data_dict)}')
    # shape=torch.Size([2, 512, 512]), dtype=torch.float32; shape=torch.Size([2, 512]), dtype=torch.bool; shape=torch.Size([2, 1024]), dtype=torch.int64; shape=torch.Size([2, 1024]), dtype=torch.bool;
    # shape=(2, 8, 512), dtype=float32; shape=(2, 8), dtype=bool; shape=(2, 16), dtype=int16; shape=(2, 16), dtype=int32; shape=(2, 16), dtype=bool;

    # Move data to device
    encoder_in = torch.as_tensor(batch_data_dict['encoder_input_tokens'], device=device)
    encoder_mask = torch.as_tensor(batch_data_dict['encoder_input_mask'], device=device)
    decoder_in = torch.as_tensor(batch_data_dict['decoder_input_tokens'], device=device, dtype=torch.int64)
    # decoder_input_ids T5自动根据labels生成
    target = torch.as_tensor(batch_data_dict['decoder_target_tokens'], device=device, dtype=torch.int64)  
    # shape=(batch, target_len); int64 即为 longTensor，t5要求的，不然实际上uint16就够了
    target_mask = torch.as_tensor(batch_data_dict['decoder_target_mask'], device=device, dtype=torch.bool)
    # 类型转换：torch.longTensor, torch.long(), torch.type(), torch.type_as()
    
    out = model(
      inputs_embeds=encoder_in, 
      attention_mask=encoder_mask, 
      decoder_input_ids=decoder_in, 
      decoder_attention_mask=target_mask
    )
    # target[target == cf.PAD_ID] = -100
    # out = model(inputs_embeds=encoder_in, attention_mask=encoder_mask, labels=target, decoder_attention_mask=target_mask)
    # loss = out.loss
    # encoder: mask都以embeds为准，ids也会先换成embeds，这里提供embeds，就不需再嵌入
    # decoder: labels右移成为input_ids，需要通过嵌入层计算成向量，默认通过静态查找表，将ids作为下标取weights一列作为目标向量: torch.nn.functional.embedding
    # weight行数必须大于ids的最大值才有得找，行数由model.json里的 "vocab_size": 6000 控制，本质就是midi事件最大编码，等于 vocab.vocab_size
    # logging.info(get_feature_desc(out))

    logits = out.logits
    # sequence_output.shape=torch.Size([2, 1024, 512]) -> lm_logits.shape=torch.Size([2, 1024, 6000])
    # 两个batch，每个由512词向量表达，通过仿射层变为6000个类别
    loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
    # logits: (2, 1024, 6000)[batch, target_len, classes] -> (2048, 6000); target: (2, 1024) -> (2048)
    # TODO: Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss = loss.item()
    epoch_loss += loss
    iteration += 1
    if iteration % 20 == 0:
      t = time.time() - begin_time
      logging.info(f'train: {epoch=}, {iteration=}, {loss=}, lr={scheduler.get_lr()}, in {t:.3f}s')
      logging.info(f'{batch_data_dict["id"]=}')
      statistics['train_loss'].append(loss)
      begin_time += t
  
  logging.info(f'-------train exits, {epoch=}-------')
  return epoch_loss / iteration


def evaluate(
  model: torch.nn.Module, 
  device: torch.device, 
  data_loader: DataLoader, 
  criterion: torch.nn.Module,
  statistics: dict
) -> float:
  # 损失函数除了作为模型训练时候的优化目标，也能够作为模型好坏的一种评价指标。但通常人们还会从其它角度评估模型的好坏，这就是评估指标。
  # 通常损失函数都可以作为评估指标，如MAE,MSE,CategoricalCrossentropy等也是常用的评估指标。
  # 但评估指标不一定可以作为损失函数，例如AUC,Accuracy,Precision。因为评估指标不要求连续可导，而损失函数通常要求连续可导。
  # TODO evaluate部分直接用交叉熵先写
    
  model.eval()
  begin_time = time.time()
  iteration = 0
  epoch_loss = 0
  epoch = data_loader._index_sampler.epoch
  logging.info(f'-------eval starts, {epoch=}-------')

  for batch_data_dict in data_loader:
    # Move data to device
    encoder_in = torch.as_tensor(batch_data_dict['encoder_input_tokens'], device=device)
    encoder_mask = torch.as_tensor(batch_data_dict['encoder_input_mask'], device=device)
    decoder_in = torch.as_tensor(batch_data_dict['decoder_input_tokens'], device=device, dtype=torch.int64)
    # decoder_input_ids T5自动根据labels生成
    target = torch.as_tensor(batch_data_dict['decoder_target_tokens'], device=device, dtype=torch.int64)  
    # shape=(batch, target_len); int64 即为 longTensor，t5要求的，不然实际上uint16就够了
    target_mask = torch.as_tensor(batch_data_dict['decoder_target_mask'], device=device, dtype=torch.bool)
    # 类型转换：torch.longTensor, torch.long(), torch.type(), torch.type_as()
    
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
      logging.info(f'eval: {epoch=}, {iteration=}, {loss=}, in {t:.3f}s')
      statistics['eval_loss'].append(loss)
      begin_time += t

  logging.info(f'-------eval exits, {epoch=}-------')
  return epoch_loss / iteration


def main(cf: YuiConfig, resume: bool=False):
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
  if 'cuda' in str(device):
      logging.info('Using GPU.')
      logging.info(f'GPU number: {torch.cuda.device_count()}')
  else:
      logging.info('Using CPU.')

  # Codec & Vocabulary
  codec = vocabularies.build_codec(cf)
  vocabulary = vocabularies.vocabulary_from_codec(codec)
    
  # Dataset
  meta_path = os.path.join(cf.DATASET_DIR, cf.DATAMETA_NAME)

  train_sampler = MaestroSampler2(meta_path, 'train', batch_size=batch_size, config=cf, max_iter_num=cf.TRAIN_ITERATION)
  train_dataset = MaestroDataset(cf.DATASET_DIR, cf, codec, vocabulary, meta_file=cf.DATAMETA_NAME)
  train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
  
  validate_sampler = MaestroSampler2(meta_path, 'validation', batch_size=batch_size, config=cf, max_iter_num=-1)
  validate_loader = DataLoader(dataset=train_dataset, batch_sampler=validate_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
  # dataset一致，主要是抽取方式sampler不同

  # Model
  t5_config = config.build_t5_config(vocab_size=vocabulary.vocab_size)
  t5_config = T5Config.from_dict(t5_config)
  model = T5ForConditionalGeneration(config=t5_config)
  # TODO 有空自己搭建
  logging.info(f'The model has {utils.count_parameters(model):,} trainable parameters')  # 15,843,712

  # Resume training
  resume_epoch = 0
  learning_rate = cf.LEARNING_RATE
  statistics = {
    'epoch': 0,
    'train_loss': [],
    'eval_loss': []
  }

  if resume:
    if not os.path.isfile(resume_checkpoint_path):
      raise FileNotFoundError(f'{resume_checkpoint_path=} is not exist')
    if os.path.isfile(statistics_path):
      statistics = torch.load(statistics_path)
      # 单独保存后面数据分析读取方便些
    checkpoint = torch.load(resume_checkpoint_path)
    # 以TRAIN_ITERATION为单位保存checkpoint
    model.load_state_dict(checkpoint['model'])
    train_sampler.load_state_dict(checkpoint['sampler'])
    validate_sampler.epoch = train_sampler.epoch
    # 二者epoch一致
    resume_epoch = checkpoint['epoch']
    learning_rate = checkpoint['learning_rate'][-1]
    # scheduler.get_lr 拿到的lr是个列表
    logging.info(f'resume training with epoch={resume_epoch}, lr={learning_rate}')

  # Parallel
  # model = torch.nn.DataParallel(model)
  # TODO 了解下再开启

  # Loss function
  criterion = torch.nn.CrossEntropyLoss(ignore_index=cf.PAD_ID)
  # 当类别==2时CrossEntropyLoss就是BCELOSS(有细微不同)，二者输入都要求(batch, class)，只是BCEloss的class=2
  # 用于处理情感分析那种输入是(n,) 输出是()的情况
  # TODO z_loss: t5x.models.BaseTransformerModel.loss_fn

  # Optimizer
  optimizer = Adafactor(model.parameters(), lr=learning_rate, scale_parameter=False, relative_step=False, warmup_init=False)
  scheduler = AdafactorSchedule(optimizer, learning_rate)
  # AdaFactor: Google提出，旨在减少显存占用，并且针对性地分析并解决了Adam的一些缺陷; 要求batch_size大一点，否则矩阵低秩分解带来的误差明显
  # 同时它会自己衰减lr，不需Schedule调整; 这里的 scheduler 只是一个取lr的代理

  model.to(device)
  best_val_loss = float('inf')
  overfit_cnt = 0
  epoch = resume_epoch
  loop_start_time = time.time()
  start_time = time.time()
  assert epoch == train_sampler.epoch, f"resume training: {epoch=} != {train_sampler.epoch=}"
  logging.info(f'-------train loop starts, {start_time=:.3f}s-------')

  # for epoch in range(resume_epoch, cf.NUM_EPOCHS):
  while epoch < cf.NUM_EPOCHS:
    train_loss = train(model, device, train_loader, criterion, optimizer, scheduler, statistics)
    current_lr = scheduler.get_lr()

    # 训练数据完整采样一轮
    if train_sampler.epoch > epoch:
      validate_sampler.reset_state()
      validate_loss = evaluate(model, device, validate_loader, criterion, statistics)
      # 等train数据完整过了一遍再进行评估
      logging.info(
        f'{epoch=} finish, time={time.time()-start_time:.3f}s, {train_loss=}, {validate_loss=}'
        f', with lr={current_lr}'
      )
      
      # Early stopping
      if validate_loss <= best_val_loss:
        best_val_loss = validate_loss
        if os.path.isfile(best_checkpoint_path):
          os.remove(best_checkpoint_path)
          # 实际上一般情况下best==resume版本，当best不存在代表resume就是最优
        overfit_cnt = 0
      elif overfit_cnt+1 < cf.OVERFIT_PATIENCE:
        if not os.path.exists(best_checkpoint_path):
          shutil.copyfile(resume_checkpoint_path, best_checkpoint_path)
          # 有过拟合倾向，将之前的resume模型作为最优保存起来
        overfit_cnt += 1
      else:
        logging.info(f'early stoping')
        break

      epoch += 1
      start_time = time.time()
      train_sampler.reset_state()
      # 重新设置sampler状态，使下一epoch切片方式有所不同
    
    # Save model
    statistics['epoch'] = epoch
    checkpoint = {
      'model': model.state_dict(),
      'sampler': train_sampler.state_dict(),
      'learning_rate': current_lr,
      'epoch': epoch,
    }
    # 上面的evaluate对train_sampler的影响也要记录下来
    # epoch要等上面计算完才能保存，保证二者记录的epoch一致
    torch.save(checkpoint, resume_checkpoint_path)
    torch.save(statistics, statistics_path)
    logging.info(f'save model and statistics to {checkpoints_dir}')
  logging.info(f'-------train loop ends, time={time.time()-loop_start_time:.3f}s-------')


if __name__ == '__main__':
  from config.data import YuiConfigDev
  cf = YuiConfigDev(
    CUDA=True
  )
  try:
    main(cf, resume=True)
    # main(cf, resume=False)
  except Exception as e:
    logging.exception(e)

  # TODO
  # `以epoch为单位保存checkpoints，修改sampler使之一个epoch产生的样本数量合适
  # 使用 tiny配置 测试 train 函数
  # 将shift固定为仅一个事件，metrics、loss应该考虑事件的构成比例
  # 添加metrics
  # 将最终metrics作为过拟合判断标准
  # 据论文，频谱计算后用dense层映射为t5的输入大小
  # `2. 优化切片逻辑，允许随机均匀切片
  # 4重写去掉tf依赖，对比原先的结果
  # 8数据增强: 训练时加入一些噪声干扰，增加健壮性
    # 像bytedance那样改变音高、考虑踏板
    # 随机切分不等长且不重叠的片段作为输入
    # 提取主旋律或统一音色后再训练
    # 随机切分不等长且可重叠的片段作为输入(需要尽量多次切片才可能覆盖整首曲子，先用基础的训练一遍再说)
  # 直接预测音符时间回归值是否比作为分类任务训练好

  # Done 
  # `0看featureconverter
  # 1构思预处理流程，不能拘泥于细节
    # `考虑mp3和wav区别 -mp3压缩了高频
    # `照着kaggle和bytedance进行预处理、做dataloader
  # `2将midi进行预处理，再测试能否正确转换回来
  # `3先按原来的featureconverter运行一遍，记录结果
  # `4.1 将np变量统一成pytorch的
  # `5数据集colab、kaggle都总会有办法，先下载下来，写好处理代码
  # 1. 写好训练部分
    # `看transformers.t5 docstring，看输入参数
    # `弄清该模型输出形式，交叉熵作为loss要求输入(batch, class)
    # `看seq2seq教程
    # `看transformer论文讲解
  # `检测loss计算时是否避开了pad部分
  # `将model配置采用python对象的方式存储，分dev和pro配置
  # `添加validation跟early stopping
  # `修改sampler的iter，在最后一首歌处理完结束，一个epoch不超过20万个样本
  # `支持训练中断与恢复，添加模型参数的保存读取
  # `记录训练过程中的loss等数据，方便画图
