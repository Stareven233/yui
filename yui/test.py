import logging
import os

from torch.utils.data import DataLoader
import note_seq

from config.data import cf
import vocabularies
import note_sequences
import preprocessors
import postprocessors
from datasets import MaestroDataset, MaestroSampler, collate_fn
from utils import create_logging


logs_dir = os.path.join(cf.WORKSPACE, 'logs')
create_logging(logs_dir, "test_", filemode='w')


def test_pre_postprocess():
  """test preprocessors and postprocessors

  以batch=4取数据，preprocess全部处理成模型输入，但不喂入模型，
  直接postprocess再将target换回ns
  比较原始的ns与变换后的ns是否一致
  处理一整首歌，验证预处理/后处理的正确性

  默认用最短的音频(45s)（即validation集里的第一个）
  """

  meta_path = os.path.join(cf.DATASET_DIR, 'maestro-v3.0.0_tiny.csv')
  sampler = MaestroSampler(meta_path, 'validation', batch_size=4, segment_second=cf.segment_second)
  dataset = MaestroDataset(cf.DATASET_DIR, cf, meta_file='maestro-v3.0.0_tiny.csv')
  # inputs.shape=(1024, 128), input_times.shape=(1024,), targets.shape=(8290,), input_event_start_indices.shape=(1024,), input_event_end_indices.shape=(1024,), input_state_event_indices.shape=(1024,), 
  data_loader = DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0, pin_memory=True)
  # 经过collate_fn处理后各特征多了一维batch_size（将batch_size个dict拼合成一个大dict）
  # inputs.shape=(4,1024, 128), input_times.shape=(4,1024,), targets.shape=(4,8290,), input_event_start_indices.shape=(4,1024,), input_event_end_indices.shape=(4,1024,), input_state_event_indices.shape=(4,1024,), 

  codec = vocabularies.build_codec(cf)
  vocabulary = vocabularies.vocabulary_from_codec(codec)
  encoding_spec = note_sequences.NoteEncodingSpec

  predictions = []
  for batch in data_loader:
    stop = False
    if stop:
      break

    for i in range(len(batch)):
      idx, start_time = eval(batch["id"][i])
      print(idx, start_time)
      exit()
      if idx != 0:
        stop = True
        break

      pred = batch["decoder_target_tokens"][i]
      pred = postprocessors.detokenize(pred, cf, vocab=vocabulary)
      # 模拟模型输出，仅有一个1-d array
      start_time -= start_time % (1 / codec.steps_per_second)

      predictions.append({
        'est_tokens': pred,
        'start_time': start_time
    })

  t = postprocessors.predictions_to_ns(predictions, codec, encoding_spec)
  logging.info(f"\n\n  for audio={meta_dict['audio_filename'][0]}")
  logging.info("ns after postprocessing:\n")
  logging.info(str(t["est_ns"]))

  meta_dict = preprocessors.read_metadata(meta_path)
  midi_path = os.path.join(cf.DATASET_DIR, meta_dict['midi_filename'][0])
  ns = note_seq.midi_file_to_note_sequence(midi_path)
  logging.info("ns after postprocessing:\n")
  logging.info(str(ns))
  try:
    logging.info(t["est_ns"] == ns)
  except Exception as e:
    logging.exception(e)
