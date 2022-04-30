from types import MappingProxyType


# setting = 'develop'
setting = 'produce'

if setting == 'develop':
  from .data import YuiConfigDev as _YuiConfig
  from .model import t5_config_dev as _t5_config
elif setting == 'produce':
  from .data import YuiConfigPro as _YuiConfig
  from .model import t5_config_pro2 as _t5_config

yui_config = _YuiConfig()
_t5_config.update(
  pad_token_id=yui_config.PAD_ID,
  eos_token_id=yui_config.ENCODED_EOS_ID,
  decoder_start_token_id=yui_config.PAD_ID,
)
t5_config = MappingProxyType(_t5_config)

def build_t5_config(**kwargs):
  """主要是根据实际的vocabulary.vocab_size设置t5的vocab_size"""

  _t5_config.update(**kwargs)
  return MappingProxyType(_t5_config)
