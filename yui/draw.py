import librosa
import matplotlib.pyplot as plt
import librosa.display
import pretty_midi

from config import yui_config
import preprocessors


def show_pianorolls(midi_file):
  pm = pretty_midi.PrettyMIDI(midi_file)
  pianorolls = pm.get_piano_roll()
  plt.figure(figsize=(12, 3))
  plt.imshow(pianorolls)
  plt.show()

def show_waveform(audio_file):
  x, sr=librosa.load(audio_file)
  plt.figure(figsize=(16, 5))
  librosa.display.waveplot(x, sr=sr)
  plt.show()


def show_spectrogram(audio_file):
  x, _=librosa.load(audio_file)
  x = preprocessors.compute_spectrograms({'inputs': x}, yui_config)
  spectrogram = x['inputs']
  plt.figure(figsize=(3, 6))
  plt.imshow(spectrogram)
  plt.show()


if __name__ == '__main__':
  audio = r'D:/Music/MuseScore/音乐/No,Thank_You.wav'
  midi = r'D:/Music/MuseScore/乐谱/No,Thank_You.mid'
  # show_waveform(audio)
  # show_spectrogram(audio)
  show_pianorolls(midi)

