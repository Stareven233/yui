# 参数见： ~\Python39\Lib\site-packages\transformers\models\t5\configuration_t5.T5Config
# 文档：https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/model#transformers.generation_utils.GenerationMixin.generate
# 实际上这里的配置仅作为参考，实际使用更多利用 config.build_t5_config 生成

t5_config_pro_full = {
  'd_model': 512,
  'd_kv': 64,  # Size of the key, query, value projections per attention head. `d_kv` has to be equal to `d_model // num_heads`.
  'd_ff': 512,
  'num_layers': 8,
  'num_decoder_layers': 8,  # Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
  'num_heads': 6,  # 一般来说要等于d_model/d_kv，但这是t5.1.1指定的
  'dropout_rate': 0.1,
  'num_beams': 4,
  'num_beam_groups': 1,
  'vocab_size': 5000,  # 应等于vocabulary.vocab_size
  'top_p': 0.95,  # 根据 https://zhuanlan.zhihu.com/p/115076102，top_p 效果比其他search/sample方法都好

  'top_k': 50,
  'feed_forward_proj': 'gated-gelu',
  'bos_token_id': 0,
  'pad_token_id': 0,
  'eos_token_id': 1,
  'sep_token_id': None,
  'decoder_start_token_id': 0,
  'forced_eos_token_id': 1,
  'transformers_version': '4.17.0',
  'model_type': 't5',
  'relative_attention_num_buckets': 32,
  'layer_norm_epsilon': 1e-06,
  'initializer_factor': 1.0,
  'use_cache': True,
  'return_dict': True,
  'output_hidden_states': False,
  'output_attentions': False,
  'torchscript': False,
  'torch_dtype': None,
  'use_bfloat16': False,
  'pruned_heads': {},
  'tie_word_embeddings': False,
  'is_encoder_decoder': True,
  'is_decoder': False,
  'cross_attention_hidden_size': None,
  'add_cross_attention': False,
  'tie_encoder_decoder': False,
  'max_length': 20,
  'min_length': 0,
  'do_sample': False,
  'early_stopping': False,
  'diversity_penalty': 0.0,
  'temperature': 1.0,
  'typical_p': 1.0,
  'repetition_penalty': 1.0,
  'length_penalty': 1.0,
  'no_repeat_ngram_size': 0,
  'encoder_no_repeat_ngram_size': 0,
  'bad_words_ids': None,
  'num_return_sequences': 1,
  'chunk_size_feed_forward': 0,
  'output_scores': False,
  'return_dict_in_generate': False,
  'forced_bos_token_id': None,
  'remove_invalid_values': False,
  'architectures': None,
  'finetuning_task': None,
  'id2label': {
    0: 'LABEL_0',
    1: 'LABEL_1'
  },
  'label2id': {
    'LABEL_0': 0,
    'LABEL_1': 1
  },
  'tokenizer_class': None,
  'prefix': None,
  'task_specific_params': None,
  'problem_type': None,
  '_name_or_path': '',
  'output_past': True
}

t5_config_dev_tiny = t5_config_pro_full | {
  # tiny model
  'd_model': 12,
  'd_kv': 3,
  'd_ff': 16,
  'num_layers': 2,
  'num_decoder_layers': 2,
  'num_heads': 4,
  'dropout_rate': 0.0,
  'num_beams': 1,
  'num_beam_groups': 1,
  'vocab_size': 5000,
}

t5_config_dev = t5_config_pro_full | {  
  'd_model': 128,  # 就处理几千个单词，根本不需要256维词嵌入吧
  'd_kv': 32,  # Size of the key, query, value projections per attention head. `d_kv` has to be equal to `d_model // num_heads`.
  'd_ff': 256,
  'num_layers': 2,
  'num_decoder_layers': 2,  # Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
  'num_heads': 4,
  'vocab_size': 4449,  # when STEPS_PER_SECOND==1000
  # 'vocab_size': 770,  # when STEPS_PER_SECOND==100
}
# 这个配置大约90万参数，能够在本地以batch_size=8训练，GPU的cuda占用率将近100%


t5_config_pro = t5_config_pro_full | {  
  'd_model': 256,  # 就处理几千个单词，根本不需要256维词嵌入吧
  'd_kv': 64,
  'd_ff': 256,
  'num_layers': 4,
  'num_decoder_layers': 4,
  'num_heads': 4,
}
# 这个配置大约500万参数，能够在本地以batch_size=4训练，GPU的cuda占用率将近100%

t5_config_pro2 = t5_config_pro_full | {  
  'd_model': 384,
  'd_kv': 64,
  'd_ff': 512,
  'num_layers': 6,
  'num_decoder_layers': 6,
  'num_heads': 6,
}
# 这个配置大约1800万参数，能够在本地以batch_size=2训练，GPU的cuda占用率将近100%

if __name__ == '__main__':
  print(t5_config_dev)
