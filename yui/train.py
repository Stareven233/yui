from base64 import decode
import os
import time
import logging

import torch
import torch.utils.data
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Config
from transformers.optimization import Adafactor

from datasets import MaestroDataset, MaestroSampler, collate_fn
from config.data import YuiConfig, cf
from utils import create_folder, create_logging, get_feature_desc


def train(cf: YuiConfig):
  """Train a piano transcription system.

  Args:
    workspace: str, directory of your workspace
    model_type: str, e.g. 'Regressonset_regressoffset_frame_velocity_CRNN'
    loss_type: str, e.g. 'regress_onset_offset_frame_velocity_bce'
    augmentation: str, e.g. 'none'
    batch_size: int
    learning_rate: float
    reduce_iteration: int
    resume_iteration: int
    early_stop: int
    device: 'cuda' | 'cpu'
    mini_data: bool
  """

  # Arugments & parameters
  workspace = cf.WORKSPACE
  batch_size = cf.BATCH_SIZE
  learning_rate = cf.LEARNING_RATE
  device = torch.device('cuda') if cf.CUDA and torch.cuda.is_available() else torch.device('cpu')
  num_workers = cf.NUM_WORKERS
  loss_type = 'adafactor'

  checkpoints_dir = os.path.join(workspace, 'checkpoints')
  create_folder(checkpoints_dir)
  logs_dir = os.path.join(workspace, 'logs')
  create_logging(logs_dir, f'train_', filemode='w', with_time=False)

  logging.info(cf)
  if 'cuda' in str(device):
      logging.info('Using GPU.')
      # Parallel
      logging.info(f'GPU number: {torch.cuda.device_count()}')
  else:
      logging.info('Using CPU.')
  
  # Model
  model = T5ForConditionalGeneration(config=T5Config.from_json_file('./yui/config/model.json'))
  # TODO 有空自己搭建

  # Dataset
  meta_path = os.path.join(cf.DATASET_DIR, cf.DATAMETA_NAME)
  train_sampler = MaestroSampler(meta_path, 'train', batch_size=batch_size, segment_second=cf.segment_second)
  train_dataset = MaestroDataset(cf.DATASET_DIR, cf, meta_file=cf.DATAMETA_NAME)
  train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

  # Loss function
  criterion = torch.nn.CrossEntropyLoss(ignore_index=cf.PAD_ID)
  # 当类别==2时CrossEntropyLoss就是BCELOSS(有细微不同)，二者输入都要求(batch, class)，只是BCEloss的class=2
  # 用于处理情感分析那种输入是(n,) 输出是()的情况
  # TODO z_loss: t5x.models.BaseTransformerModel.loss_fn

  # Optimizer
  optimizer = Adafactor(model.parameters(), lr=1e-3, scale_parameter=False, relative_step=False, warmup_init=False)
  # AdaFactor: Google提出，旨在减少显存占用，并且针对性地分析并解决了Adam的一些缺陷; 要求batch_size大一点，否则矩阵低秩分解带来的误差明显
  # 同时它会自己衰减lr，不需Schedule调整

  # Resume training
  # TODO
  
  # model = torch.nn.DataParallel(model)
  # TODO 了解下再开启
  print(str(device))
  if 'cuda' in str(device):
    model.to(device)
  model.train()

  train_bgn_time = time.time()
  iteration = 1

  for batch_data_dict in train_loader:
    # logging.info(f'{get_feature_desc(batch_data_dict)}')
    # shape=torch.Size([2, 512, 512]), dtype=torch.float32; shape=torch.Size([2, 512]), dtype=torch.bool; shape=torch.Size([2, 1024]), dtype=torch.int64; shape=torch.Size([2, 1024]), dtype=torch.bool;
    
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
    logging.info(get_feature_desc(out))

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

    
    if iteration % 100 == 0:
      t = time.time() - train_bgn_time
      logging.info(f'{iteration=}, {loss=}, lr={optimizer.param_groups["lr"]}, in {t:.3f}s')
      train_bgn_time += t
      # iteration=0, loss=tensor(8.9052, grad_fn=<NllLossBackward0>)

    # Save model
    # if iteration % 20000 == 0:
    #   checkpoint = {
    #     'iteration': iteration, 
    #     # 'model': model.module.state_dict(), 
    #     'sampler': train_sampler.state_dict()
    #   }
    #   checkpoint_path = os.path.join(checkpoints_dir, f'{iteration}_iterations.pth')

    #   torch.save(checkpoint, checkpoint_path)
    #   logging.info(f'Model saved to {checkpoint_path}')

    if iteration == cf.TRAIN_EPOCHS:
      break
    iteration += 1


if __name__ == '__main__':
  train(cf)

  # TODO
  # 1. 写好训练部分
    # `看transformers.t5 docstring，看输入参数
    # `弄清该模型输出形式，交叉熵作为loss要求输入(batch, class)
    # 看seq2seq教程
    # 看transformer论文讲解
  # 3. 优化train逻辑，支持epoch
  # 添加validation跟early stopping
  # 支持训练中断与恢复，添加模型参数的保存读取
  # 添加metrics
  # 将model配置采用python对象的方式存储
  # 2. 优化切片逻辑，允许随机均匀切片
  # 6优化/减小模型，降低内存、运行时间
  # 4重写去掉tf依赖，对比原先的结果
  # 8数据增强
    # 像bytedance那样改变音高、考虑踏板
    # 随机切分不等长且允许重叠的片段作为输入
    # 提取主旋律或统一音色后再训练

  # Done 
  # `0看featureconverter
  # 1构思预处理流程，不能拘泥于细节
    # `考虑mp3和wav区别 -mp3压缩了高频
    # `照着kaggle和bytedance进行预处理、做dataloader
  # `2将midi进行预处理，再测试能否正确转换回来
  # `3先按原来的featureconverter运行一遍，记录结果
  # `4.1 将np变量统一成pytorch的
  # `5数据集colab、kaggle都总会有办法，先下载下来，写好处理代码
