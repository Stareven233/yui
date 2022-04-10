import logging
import os

from torch.utils.data import DataLoader
import note_seq

from config.data import YuiConfigPro
import vocabularies
import note_sequences
import preprocessors
import postprocessors
from datasets import MaestroDataset, MaestroSampler, collate_fn, MaestroSampler2
from utils import create_logging


# cf = YuiConfigPro(
#   DATASET_DIR=r'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0/',
#   DATAMETA_NAME=r'maestro-v3.0.0_tiny.csv',
#   WORKSPACE=r'D:/A日常/大学/毕业设计/code/yui/',
# )
cf = YuiConfigPro(
  BATCH_SIZE=32,  # 一个核16个
  NUM_WORKERS=2,
  NUM_EPOCHS=2,
  DATASET_DIR=r'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0/',
  DATAMETA_NAME=r'maestro-v3.0.0_tiny.csv',
  WORKSPACE=r'D:/A日常/大学/毕业设计/code/yui/',
)
logs_dir = os.path.join(cf.WORKSPACE, 'logs')
create_logging(logs_dir, "test_", filemode='w')

logging.info(cf)


def test_pre_postprocess():
  """test preprocessors and postprocessors

  以batch=4取数据，preprocess全部处理成模型输入，但不喂入模型，
  直接postprocess再将target换回ns
  比较原始的ns与变换后的ns是否一致
  处理一整首歌，验证预处理/后处理的正确性

  默认用最短的音频(45s)（即validation集里的第一个）
  """

  meta_path = os.path.join(cf.DATASET_DIR, 'maestro-v3.0.0_tiny.csv')
  codec = vocabularies.build_codec(cf)
  
  vocabulary = vocabularies.Vocabulary(cf, codec.num_classes, extra_ids=cf.EXTRA_IDS)
  # logging.info(codec.num_classes)  # 4611
  # logging.info(vocabulary.vocab_size)  # 4714

  sampler = MaestroSampler(meta_path, 'validation', batch_size=4, segment_second=cf.segment_second)
  dataset = MaestroDataset(cf.DATASET_DIR, cf, codec, vocabulary,  meta_file='maestro-v3.0.0_tiny.csv')
  # inputs.shape=(1024, 128), input_times.shape=(1024,), targets.shape=(8290,), input_event_start_indices.shape=(1024,), input_event_end_indices.shape=(1024,), input_state_event_indices.shape=(1024,), 
  data_loader = DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0, pin_memory=True)
  # 经过collate_fn处理后各特征多了一维batch_size（将batch_size个dict拼合成一个大dict）
  # inputs.shape=(4,1024, 128), input_times.shape=(4,1024,), targets.shape=(4,8290,), input_event_start_indices.shape=(4,1024,), input_event_end_indices.shape=(4,1024,), input_state_event_indices.shape=(4,1024,), 

  encoding_spec = note_sequences.NoteEncodingSpec
  predictions = []
  stop = False

  for batch in data_loader:
    if stop:
      break

    print('get a new batch')
    for i in range(len(batch["id"])):
      idx, start_time = eval(batch["id"][i])
      print(f'get {idx=} and {start_time=}')
      if idx > 0:
        stop = True
        break

      pred = batch["decoder_target_tokens"][i]
      pred = postprocessors.detokenize(pred, cf, vocab=vocabulary)
      # 模拟模型输出，仅有一个1-d array
      # start_time -= start_time % (1 / codec.steps_per_second)
      # 当input_length=512时切片一段为4.096s，精确到毫秒，如果执行上述语句
      # 则去掉了末尾，变为4.09，精确到10ms级别，造成该片段前后重叠

      predictions.append({
        'est_tokens': pred,
        'start_time': start_time
    })
    # predictions.append((start_time, pred))

  t = postprocessors.predictions_to_ns(predictions, codec, encoding_spec)
  est_ns = t["est_ns"]

  meta_dict = preprocessors.read_metadata(meta_path)
  midi_file = os.path.join(cf.DATASET_DIR, meta_dict['midi_filename'][0])
  logging.info(f'get origin {midi_file=}')
  logging.info(f"for {midi_file=}, ns after postprocessing:\n")

  ns = note_seq.midi_file_to_note_sequence(midi_file)
  new_midi_file = midi_file.removesuffix(".midi") + "_processed.midi"
  
  # setattr(sequence, "control_changes", ns.control_changes)
  # AttributeError: Assignment not allowed to repeated field "control_changes" in protocol message object.
  note_seq.sequence_proto_to_midi_file(est_ns, new_midi_file)
  logging.info(f"output processed midi file to {new_midi_file}")
  # midi_file_to_note_sequence, sequence_proto_to_midi_file至少一个有问题
  # 好好一个midi输出后再读入发现start_time、end_time小数位精度只有2位，离谱
  # 同一个midi读入输出结果一直在变，1.06 -> 1.059090909090909, 这应该也是谱子变得离谱

  # logging.info(f"origin {midi_file=}:\n {ns}\n")
  # logging.info(f"processed {new_midi_file=}:\n {est_ns}\n")
  
  try:
    logging.info(est_ns == ns)
  except Exception as e:
    logging.exception(e)

  # 测试发现仅是max_input_length不同，乐谱(est_ns, ns)看起来就差异很大，然而听起来却差不多
  # 打印note_sequences比较发现二者长度一样，pitch跟velocity都一样，就是开始结束时间细微不同，毫秒级别


def test_datasets(meta: tuple[int, float]):
  """test datasets, sampler and dataloader

  以batch=1取数据，并输出取到的结果，使用logging记录过程中的info
  """

  codec = vocabularies.build_codec(cf)
  
  vocabulary = vocabularies.Vocabulary(cf, codec.num_classes, extra_ids=cf.EXTRA_IDS)
  dataset = MaestroDataset(cf.DATASET_DIR, cf, codec, vocabulary,  meta_file='maestro-v3.0.0_tiny.csv')
  # inputs.shape=(1024, 128), input_times.shape=(1024,), targets.shape=(8290,), input_event_start_indices.shape=(1024,), input_event_end_indices.shape=(1024,), input_state_event_indices.shape=(1024,), 
  dataset[meta]
  

def test_dataloader():
  """test datasets, sampler and dataloader

  以batch=1取数据，并输出取到的结果，使用logging记录过程中的info
  """

  meta_path = os.path.join(cf.DATASET_DIR, 'maestro-v3.0.0_tiny.csv')
  codec = vocabularies.build_codec(cf)
  
  vocabulary = vocabularies.Vocabulary(cf, codec.num_classes, extra_ids=cf.EXTRA_IDS)
  sampler = MaestroSampler2(meta_path, 'test', batch_size=16, config=cf, max_iter_num=-1, drop_last=True)
  dataset = MaestroDataset(cf.DATASET_DIR, cf, codec, vocabulary,  meta_file='maestro-v3.0.0_tiny.csv')
  # inputs.shape=(1024, 128), input_times.shape=(1024,), targets.shape=(8290,), input_event_start_indices.shape=(1024,), input_event_end_indices.shape=(1024,), input_state_event_indices.shape=(1024,), 
  data_loader = DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0, pin_memory=True)
  # 经过collate_fn处理后各特征多了一维batch_size（将batch_size个dict拼合成一个大dict）
  # inputs.shape=(4,1024, 128), input_times.shape=(4,1024,), targets.shape=(4,8290,), input_event_start_indices.shape=(4,1024,), input_event_end_indices.shape=(4,1024,), input_state_event_indices.shape=(4,1024,), 

  it = iter(data_loader)
  features = next(it)
  logging.info(f'{features=}')

  # midi_file = os.path.join(cf.DATASET_DIR, '2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--3_preprocessed.midi')
  # note_seq.note_sequence_to_midi_file(ns, midi_file)


def test_midi_diff(data_dir, midi_file):
  # attr = {'time_signatures', 'key_signatures', 'tempos', 'instrument_infos', 'notes'}
    # pretty_midi.PrettyMIDI(midi)
  ns = note_seq.midi_file_to_note_sequence(os.path.join(data_dir, midi_file))
  logging.info(f"for {midi_file=}, \n{ns=}")


if __name__ == '__main__':
  try:
    # test_datasets((0, 44.384))

    test_dataloader()

    # test_pre_postprocess()

    # midi = 'MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--3.midi'
    # midi = 'MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--3_processeds100.midi'
    # midi = 'MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--3_processeds1000.midi'
    # midi = 'MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--3_processeds999.midi'
    # test_midi_diff(r'D:/A日常/大学/毕业设计/dataset/maestro-v3.0.0/2015', midi)
  except Exception as e:
    logging.exception(e)
