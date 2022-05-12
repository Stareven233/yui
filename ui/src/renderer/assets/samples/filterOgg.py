"""remove all files in other format except .ogg"""

import os

for inst in os.listdir('.'):
  print(f'processing {inst=}')
  inst_dir = f'./{inst}'
  if(not os.path.isdir(inst_dir)):
    continue
  for file in os.listdir(inst_dir):
    if(os.path.splitext(file)[1] == '.ogg'):
      continue
    os.remove(os.path.join(inst_dir, file))
  print('finish')
