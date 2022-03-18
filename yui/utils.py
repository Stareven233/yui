import os
import logging
from datetime import datetime

import numpy as np


def get_feature_desc(f:dict[str, np.ndarray]):
  desc = ''
  if not isinstance(f, dict):
    return f'{str(f)=}'
  for k, v in f.items():
    desc += f'{k}, {type(v)}, '
    if isinstance(v, np.ndarray):
      desc += f'shape={v.shape}; ' 
    elif isinstance(v, (list, tuple,)):
      desc += f'len={len(v)}; '
    else:
      desc += f'{str(v)=}'
  return desc


class Namespace:
  def __init__(self, **kwargs):
    for name in kwargs:
      setattr(self, name, kwargs[name])

  def __eq__(self, other):
    if not isinstance(other, Namespace):
      return NotImplemented
    return vars(self) == vars(other)

  def __contains__(self, key):
    return key in self.__dict__


def create_folder(fd):
  if not os.path.exists(fd):
    os.makedirs(fd)


def get_filename(path):
  path = os.path.realpath(path)
  name_ext = path.split('/')[-1]
  name = os.path.splitext(name_ext)[0]
  return name


def create_logging(
  log_dir, 
  prefix='', 
  filemode='w', 
  level=logging.INFO,
  print_console=True
):
  create_folder(log_dir)
  # 用DEBUG调试的话会有许多numpy?的输出

  name = prefix + datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f') 
  log_path = os.path.join(log_dir, f'{name}.log')
  logging.basicConfig(
    level=level,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=log_path,
    filemode=filemode
  )

  # Print to console
  if print_console:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-6s: %(levelname)-4s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

  return logging


def float32_to_int16(x):
  assert np.max(np.abs(x)) <= 1.
  return (x * 32767.).astype(np.int16)


def int16_to_float32(x):
  return (x / 32767.).astype(np.float32)
