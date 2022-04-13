import datasets
import event_codec
import note_sequences
import preprocessors
import vocabularies
import utils
import postprocessors
import config
"""
为了白嫖kaggle，不得已整一个py37版本的代码
主要修改：
1. 3.8 海象运算符
2. 3.8 f-string里的=号
3. 3.9 类型注解里使用的原生类型
4. 3.9 字典合并用的 |
"""

__all__ = [
  'datasets',
  'event_codec',
  'note_sequences',
  'preprocessors',
  'vocabularies',
  'utils',
  'postprocessors',
  'config',
]


