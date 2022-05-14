"""train
模型训练
"""

import code
import os
import time
import logging
import math

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Config
from transformers.optimization import Adafactor, AdafactorSchedule

from datasets import MaestroDataset3, MaestroSampler2, collate_fn
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
  epoch_loss = 0
  epoch = data_loader._index_sampler.epoch
  st = time.time()
  verbose_gap = max(accumulation_steps, 50)
  logging.debug(f'-------train starts, {epoch=}-------')

  for batch_data_dict in data_loader:
    # colab上10分钟准备不好128个sample，io问题很大
    # logging.debug(f'{get_feature_desc(batch_data_dict)}')
    # shape=torch.Size([2, 512, 512]), dtype=torch.float32; shape=torch.Size([2, 512]), dtype=torch.bool; shape=torch.Size([2, 1024]), dtype=torch.int64; shape=torch.Size([2, 1024]), dtype=torch.bool;
    # shape=(2, 8, 512), dtype=float32; shape=(2, 8), dtype=bool; shape=(2, 16), dtype=int16; shape=(2, 16), dtype=int32; shape=(2, 16), dtype=bool;

    # Move data to device    
    logging.debug(f'-------train, {iteration=}-------')
    encoder_in, encoder_mask, decoder_in, target, target_mask = utils.move_to_device(batch_data_dict, device)
    logging.debug(f'data: {time.time()-st}')

    st = time.time()
    out = model(
      inputs_embeds=encoder_in, 
      attention_mask=encoder_mask, 
      decoder_input_ids=decoder_in, 
      decoder_attention_mask=target_mask
    )
    # encoder: mask都以embeds为准，ids也会先换成embeds，这里提供embeds，就不需再嵌入
    # decoder: labels右移成为input_ids，需要通过嵌入层计算成向量，默认通过静态查找表，将ids作为下标取weights一列作为目标向量: torch.nn.functional.embedding
    # weight行数必须大于ids的最大值才有得找，行数由model.json里的 "vocab_size": 6000 控制，本质就是midi事件最大编码，等于 vocab.vocab_size
    # attention_mask 文档疑似有误，如try.py所示，取值仅在0/1两个取，而且pad统一填充在右边

    logits = out.logits
    del out
    # sequence_output.shape=torch.Size([2, 1024, 512]) -> lm_logits.shape=torch.Size([2, 1024, 6000])
    # 两个batch，每个由512词向量表达，通过仿射层变为6000个类别
    # utils.trunc_logits_by_eos(logits)
    loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1)) / accumulation_steps
    # logits: (2, 1024, 6000)[batch, target_len, classes] -> (2048, 6000); target: (2, 1024) -> (2048)
    # 也是因为 accumulation_steps 的存在，更改batch_size会造成loss数值的改变

    iteration += 1
    # Backward
    loss.backward()
    if iteration % accumulation_steps == 0:
      optimizer.step()
      optimizer.zero_grad()
      # 梯度累加 gradient accumulation

    loss = loss.item()
    epoch_loss += loss
    if iteration % verbose_gap == 0:  #  and iteration > accumulation_steps
      t = time.time() - begin_time
      logging.info(f'train: {epoch=}, {iteration=}, {loss=}, lr={scheduler.get_lr()}, in {t:.3f}s')
      # utils.show_pred(logits[0], target[0], target_mask[0])
      # logging.debug(f'id={batch_data_dict["id"].tolist()}')
      begin_time += t
    
    logging.debug(f'model: {time.time()-st}')
    st = time.time()
  
  logging.info(f'-------train exits, {epoch=}-------')
  return epoch_loss / iteration


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
  epoch_avg_loss = 0
  epoch = data_loader._index_sampler.epoch
  logging.info(f'-------eval starts, {epoch=}-------')

  for batch_data_dict in data_loader:
    # Move data to device
    encoder_in, encoder_mask, decoder_in, target, target_mask = utils.move_to_device(batch_data_dict, device)
    out = model(
      inputs_embeds=encoder_in, 
      attention_mask=encoder_mask, 
      decoder_input_ids=decoder_in, 
      decoder_attention_mask=target_mask
    )

    logits = out.logits
    del out
    loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
    loss = loss.item()
    epoch_avg_loss = (epoch_avg_loss*(iteration - 1) + loss) / iteration

    iteration += 1
    if iteration % 50 == 0:
      t = time.time() - begin_time
      logging.info(f'eval: {epoch=}, {iteration=}, {loss=}, in {t:.3f}s')
      begin_time += t

  logging.info(f'-------eval exits, {epoch=}-------')
  return epoch_avg_loss


class Adafactor2(Adafactor):
  def __init__(
    self,
    params,
    lr=None,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    scale_parameter=True,
    relative_step=True,
    warmup_init=False,
  ):
    super().__init__(params, lr, eps, clip_threshold, decay_rate, beta1, weight_decay, scale_parameter, relative_step, warmup_init)
    self._lr_peak_step = 1500

  @staticmethod
  def _get_lr(param_group, param_state):
    rel_step_sz = param_group["lr"]
    if param_group["relative_step"]:
      min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-3
      exp_lr = math.exp(-(6.45 + param_state["step"] / 3e4))
      # 这个值将在step=[1500,30000]从1.5e-3降到9.6e-4
      rel_step_sz = min(min_step, exp_lr)
    if param_group["scale_parameter"]:
      rel_step_sz *= max(param_group["eps"][1], param_state["RMS"])
    return rel_step_sz


def main(cf: YuiConfig, t5_config: T5Config, codec, vocabulary, resume: bool=False):
  # Arugments & parameters
  batch_size = cf.BATCH_SIZE
  device = torch.device('cuda') if cf.CUDA and torch.cuda.is_available() else torch.device('cpu')
  num_workers = cf.NUM_WORKERS

  # Checkpoint & Log
  checkpoints_dir = os.path.join(cf.WORKSPACE, 'checkpoints')
  utils.create_folder(checkpoints_dir)
  logs_dir = os.path.join(cf.WORKSPACE, 'logs')
  utils.create_logging(logs_dir, f'train', filemode='w', with_time=True)
  resume_checkpoint_path = os.path.join(checkpoints_dir, f'model_resume{cf.MODEL_SUFFIX}.pt')
  best_checkpoint_path = os.path.join(checkpoints_dir, f'model_best{cf.MODEL_SUFFIX}.pt')
  statistics_path = os.path.join(checkpoints_dir, f'statistics{cf.MODEL_SUFFIX}.pt')

  logging.info(cf)
  if device.type == 'cuda':
    logging.info('Using GPU.')
    logging.info(f'GPU number: {torch.cuda.device_count()}')
  else:
    logging.info('Using CPU.')

  # Dataset
  meta_path = os.path.join(cf.DATASET_DIR, cf.DATAMETA_NAME)

  train_sampler = MaestroSampler2(meta_path, 'train', batch_size=batch_size, config=cf, max_iter_num=cf.max_iter_num, loop=True)
  train_dataset = MaestroDataset3(cf.DATASET_DIR, cf, codec, vocabulary, meta_file=cf.DATAMETA_NAME)
  train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=False)
  
  validate_sampler = MaestroSampler2(meta_path, 'validation', batch_size=batch_size*3, config=cf, max_iter_num=-1)
  validate_loader = DataLoader(dataset=train_dataset, batch_sampler=validate_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=False)
  # pin_memory: 锁页内存，不会与虚存进行交换，转到gpu时快一些，但很容易超出gpu显存
  # dataset一致，主要是抽取方式sampler不同

  # Model
  t5_config = T5Config.from_dict(t5_config)
  logging.info(t5_config)
  model = T5ForConditionalGeneration(config=t5_config)
  logging.info(f'The model has {model.num_parameters():,} trainable parameters')
  # 17,896 for dev; 48,626,048 for pro; while T5-Small has 60 million parameters
  return
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
  statistics = {
    'epoch': 0,
    'train_loss': [],
    'eval_loss': []
  }

  # Loss function
  criterion = torch.nn.CrossEntropyLoss(ignore_index=cf.PAD_ID)
  # 当类别==2时CrossEntropyLoss就是BCELOSS(有细微不同)，二者输入都要求(batch, class)，只是BCEloss的class=2
  # 用于处理情感分析那种输入是(n,) 输出是()的情况
  # TODO: Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
  # z_loss: t5x.losses.cross_entropy_with_logits

  # Optimizer
  # optimizer = torch.optim.AdamW(model.parameters(), lr=cf.LEARNING_RATE)
  # scheduler = utils.DummySchedule(cf.LEARNING_RATE)
  # optimizer = Adafactor(model.parameters(), lr=cf.LEARNING_RATE, scale_parameter=False, relative_step=False, warmup_init=False)
  optimizer = Adafactor2(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
  scheduler = AdafactorSchedule(optimizer, cf.LEARNING_RATE)
  # AdaFactor: Google提出，旨在减少显存占用，并且针对性地分析并解决了Adam的一些缺陷; 要求batch_size大一点，否则矩阵低秩分解带来的误差明显
  # 同时它会自己衰减lr，不需Schedule调整; 这里的 scheduler 只是一个取lr的代理

  # Parallel
  # model = torch.nn.DataParallel(model)
  # 需要多张GPU并行计算，而且加载模型评估、推断时似乎也要求多卡

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
    # scheduler.get_lr 拿到的lr是个列表
    optimizer.load_state_dict(checkpoint['optimizer'])
    logging.info(f'resume training with epoch={resume_epoch}')

  # print(len(optimizer.param_groups))
  # print(optimizer.param_groups[0].keys())
  # params = optimizer.param_groups[0]
  # logging.info(optimizer.state)
  # lrs = [
  #     optimizer._get_lr(group, optimizer.state[group["params"][0]])
  #     for group in optimizer.param_groups
  # ]
  # logging.info(lrs)
  # exit()

  epoch = resume_epoch
  loop_start_time = time.time()
  start_time = time.time()
  # assert epoch == train_sampler.epoch, f"resume training: {epoch=} != {train_sampler.epoch=}"
  logging.info(f'-------train loop starts, {start_time=:.3f}s-------')

  # for epoch in range(resume_epoch, cf.NUM_EPOCHS):
  while epoch < cf.NUM_EPOCHS:
    optimizer.zero_grad()
    train_loss = train(model, device, train_loader, criterion, optimizer, scheduler, accumulation_steps=cf.accumulation_steps)
    statistics['train_loss'].append(train_loss)
    # current_lr = scheduler.get_lr()

    # 训练数据完整采样一轮
    # if train_sampler.epoch > epoch:
      # validate_sampler.reset_state()
      # validate_loss = evaluate(model, device, validate_loader, criterion)
      # exit()
      # statistics['eval_loss'].append(train_loss)
      # # 等train数据完整过了一遍再进行评估
      # logging.info(
      #   f'{epoch=} finish, time={time.time()-start_time:.3f}s, {train_loss=}, {validate_loss=}'
      #   f', with lr={current_lr}'
      # )

      # early_stopping(validate_loss)
      # if early_stopping.stop:
      #   logging.info(f'early stoping')
      #   break

      # epoch += 1
      # start_time = time.time()
      # train_sampler.reset_state()
      # # 重新设置sampler状态，使下一epoch切片方式有所不同

    # Save model
    statistics['epoch'] = epoch
    checkpoint = {
      'epoch': epoch,
      'model': model.state_dict(),
      'sampler': train_sampler.state_dict(),
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


  cf = YuiConfigDev(
    # MAX_TARGETS_LENGTH=512,
    DATASET_DIR=r'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0_hdf5/',
    # DATASET_DIR=r'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0/',
    DATAMETA_NAME=r'maestro-v3.0.0.csv',
    # DATAMETA_NAME=r'maestro-v3.0.0_tinymp3.csv',
    WORKSPACE=r'D:/A日常/大学/毕业设计/code/yui/',

    BATCH_SIZE=2,
    NUM_EPOCHS=20000,
    NUM_MEL_BINS=384,
    MODEL_SUFFIX='_autodl',
    TRAIN_ITERATION=1200,
  )
  # 用于本地测试的pro配置

  codec = vocabularies.build_codec(cf)
  vocabulary = vocabularies.Vocabulary(cf, codec.num_classes, extra_ids=cf.EXTRA_IDS)
  
  t5_config = config.build_t5_config(
    d_model=cf.NUM_MEL_BINS,
    vocab_size=vocabulary.vocab_size,
    max_length=cf.MAX_TARGETS_LENGTH,
  )

  try:
    # main(cf, t5_config, codec, vocabulary, resume=True)
    main(cf, t5_config, codec, vocabulary, resume=False)
  except Exception as e:
    logging.exception(e)

  # TODO list
  # 8数据增强: -也可写入论文当做自己的 
  # 训练时加入一些噪声干扰，增加健壮性
    # 像bytedance那样改变音高、考虑踏板
    # 提取主旋律或统一音色后再训练
  # 直接预测音符时间回归值是否比作为分类任务训练好
  # 尝试使用 LongT5
  # 输出token作为多分类任务处理(5000类左右) CEloss 似乎仍可行 -> 换成 circle loss 试试

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
  # `2. 优化切片逻辑，允许随机均匀切片
  # `记录训练过程中的loss等数据，方便画图
  # `以epoch为单位保存checkpoints，修改sampler使之一个epoch产生的样本数量合适
  # `使用 tiny配置 测试 train 函数
  # `添加metrics
  # `整理train.evaluate，不计算metrics，尽快训练;
  # `测试metrics
  # `确认模型其他参数、todo
  # `4重写去掉tf依赖，对比原先的结果
  # `mel_spectrom 尺寸会变化 -修改切片逻辑
  # `mp3读取缓慢 -使用pydub 且 为dataset增加cache
  # `为 kaggle 修改 meta csv 表，省去复制数据集操作
  # `测量 io时间跟模型推断时间 -midi 缺少 cache
  # `测试不同batchsize能否resume
  # `target_len=1024仍然不足 -> dataset里检查超出则剪短该次输入; 之后再更改len得重新训练
  # `将dataset里的cache改进为队列存储，防止多线程数据读取抖动
  # `datasets.sampler 当中断时曲子恰好处理完 sample_list[0] IndexError
  # `尝试warm-up -神奇般地收敛了！
  # `多次backward再进行optim.step - 梯度累加
  # `adafactor换成adam试试 -似乎是可以收敛，但epoch=71才到4.62真的慢
  # `logits里eos出现得太早，而且交叉熵未对这种现象施加惩罚 -但不应有影响
  # `或许应尝试减小shift，从而减少总类别 -理论上有效，类别数少的话每个类的样本相对就更多一些
  # `用将整个数据集处理成h5py格式
  # `eval, infer 时使用 model.generate + top_p sample
