import dataclasses
import itertools
import logging
from typing import MutableMapping, MutableSet, Optional, Sequence, Tuple

import note_seq

import event_codec
import run_length_encoding
import vocabularies


DEFAULT_VELOCITY = 100
DEFAULT_NOTE_DURATION = 0.01

# Quantization can result in zero-length notes; enforce a minimum duration.
MIN_NOTE_DURATION = 0.01
# 音符最短持续时间，单位为秒，等于这里设置的编码解码分辨率STEPS_PER_SECOND


@dataclasses.dataclass
class TrackSpec:
  name: str
  program: int = 0
  is_drum: bool = False


def extract_track(ns, program, is_drum):
  track = note_seq.NoteSequence(ticks_per_quarter=220)
  track_notes = [note for note in ns.notes
                 if note.program == program and note.is_drum == is_drum]
  track.notes.extend(track_notes)
  track.total_time = (max(note.end_time for note in track.notes)
                      if track.notes else 0.0)
  return track


def trim_overlapping_notes(ns: note_seq.NoteSequence) -> note_seq.NoteSequence:
  """Trim overlapping notes from a NoteSequence, dropping zero-length notes."""
  ns_trimmed = note_seq.NoteSequence()
  ns_trimmed.CopyFrom(ns)
  channels = set((note.pitch, note.program, note.is_drum)
                 for note in ns_trimmed.notes)
  for pitch, program, is_drum in channels:
    notes = [note for note in ns_trimmed.notes if note.pitch == pitch
             and note.program == program and note.is_drum == is_drum]
    sorted_notes = sorted(notes, key=lambda note: note.start_time)
    for i in range(1, len(sorted_notes)):
      if sorted_notes[i - 1].end_time > sorted_notes[i].start_time:
        sorted_notes[i - 1].end_time = sorted_notes[i].start_time
  valid_notes = [note for note in ns_trimmed.notes
                 if note.start_time < note.end_time]
  del ns_trimmed.notes[:]
  ns_trimmed.notes.extend(valid_notes)
  return ns_trimmed


def assign_instruments(ns: note_seq.NoteSequence) -> None:
  """Assign instrument numbers to notes; modifies NoteSequence in place."""

  program_instruments = dict()
  for note in ns.notes:
    if note.program not in program_instruments and not note.is_drum:
      num_instruments = len(program_instruments)
      # TODO 按出现顺序给编码？感觉很离谱
      note.instrument = (num_instruments if num_instruments < 9 else num_instruments + 1)
      program_instruments[note.program] = note.instrument
    elif note.is_drum:
      note.instrument = 9
    else:
      note.instrument = program_instruments[note.program]


def validate_note_sequence(ns: note_seq.NoteSequence) -> None:
  """Raise ValueError if NoteSequence contains invalid notes."""
  for note in ns.notes:
    if note.start_time >= note.end_time:
      raise ValueError('note has start time >= end time: %f >= %f' %
                       (note.start_time, note.end_time))
    if note.velocity == 0:
      raise ValueError('note has zero velocity')


def note_arrays_to_note_sequence(
    onset_times: Sequence[float],
    pitches: Sequence[int],
    offset_times: Optional[Sequence[float]] = None,
    velocities: Optional[Sequence[int]] = None,
    programs: Optional[Sequence[int]] = None,
    is_drums: Optional[Sequence[bool]] = None
) -> note_seq.NoteSequence:
  """Convert note onset / offset / pitch / velocity arrays to NoteSequence."""
  ns = note_seq.NoteSequence(ticks_per_quarter=220)
  for onset_time, offset_time, pitch, velocity, program, is_drum in itertools.zip_longest(
      onset_times, [] if offset_times is None else offset_times,
      pitches, [] if velocities is None else velocities,
      [] if programs is None else programs,
      [] if is_drums is None else is_drums):
    if offset_time is None:
      offset_time = onset_time + DEFAULT_NOTE_DURATION
    if velocity is None:
      velocity = DEFAULT_VELOCITY
    if program is None:
      program = 0
    if is_drum is None:
      is_drum = False
    ns.notes.add(
        start_time=onset_time,
        end_time=offset_time,
        pitch=pitch,
        velocity=velocity,
        program=program,
        is_drum=is_drum)
    ns.total_time = max(ns.total_time, offset_time)
  assign_instruments(ns)
  return ns


@dataclasses.dataclass
class NoteEventData:
  pitch: int
  velocity: Optional[int] = None
  program: Optional[int] = None
  is_drum: Optional[bool] = None
  instrument: Optional[int] = None

# TODO 实际上后3个都没用，NoteEventData(pitch=48, velocity=96, program=0, is_drum=False, instrument=None)


def note_sequence_to_onsets(
  ns: note_seq.NoteSequence
) -> Tuple[Sequence[float], Sequence[NoteEventData]]:
  """Extract note onsets and pitches from NoteSequence proto."""
  # Sort by pitch to use as a tiebreaker for subsequent stable sort.
  notes = sorted(ns.notes, key=lambda note: note.pitch)
  return ([note.start_time for note in notes],
          [NoteEventData(pitch=note.pitch) for note in notes])


def note_sequence_to_onsets_and_offsets(
  ns: note_seq.NoteSequence,
) -> Tuple[Sequence[float], Sequence[NoteEventData]]:
  """Extract onset & offset times and pitches from a NoteSequence proto.

  The onset & offset times will not necessarily be in sorted order.

  Args:
    ns: NoteSequence from which to extract onsets and offsets.

  Returns:
    times: A list of note onset and offset times.
    values: A list of NoteEventData objects where velocity is zero for note
        offsets.
  """
  # Sort by pitch and put offsets before onsets as a tiebreaker for subsequent
  # stable sort.
  notes = sorted(ns.notes, key=lambda note: note.pitch)
  times = ([note.end_time for note in notes] + [note.start_time for note in notes])
  values = (
    [NoteEventData(pitch=note.pitch, velocity=0) for note in notes] +
    [NoteEventData(pitch=note.pitch, velocity=note.velocity) for note in notes]
  )
  return times, values


def note_sequence_to_onsets_and_offsets_and_programs(
  ns: note_seq.NoteSequence,
) -> Tuple[Sequence[float], Sequence[NoteEventData]]:
  """Extract onset & offset times and pitches & programs from a NoteSequence.

  The onset & offset times will not necessarily be in sorted order.

  Args:
    ns: NoteSequence from which to extract onsets and offsets.

  Returns:
    times: A list of note onset and offset times.
    values: A list of NoteEventData objects where velocity is zero for note offsets.
  """

  # Sort by program and pitch and put offsets before onsets as a tiebreaker for
  # subsequent stable sort.
  notes = sorted(ns.notes, key=lambda note: (note.is_drum, note.program, note.pitch))
  # TODO maestro都是钢琴曲，这里is_drum, program都没什么意义

  times = (
    [note.end_time for note in notes if not note.is_drum] +
    [note.start_time for note in notes]
  )
  # midi中记录了音符的开始结束时间
  values = (
    [NoteEventData(pitch=note.pitch, velocity=0, program=note.program, is_drum=False) for note in notes if not note.is_drum] +
    [NoteEventData(pitch=note.pitch, velocity=note.velocity, program=note.program, is_drum=note.is_drum) for note in notes]
  )
  # 时间与音符事件一一对应，结束事件用同音高但力度0来表示，后面encode_events再根据times排序结果对values排序，
  # 从而将音符结束事件也当做一般的音符对待：遇到velocity=0的事件就说明前面对应的音符事件结束
  return times, values


@dataclasses.dataclass
class NoteEncodingState:
  """Encoding state for note transcription, keeping track of active pitches."""
  # velocity bin for active pitches and programs
  active_pitches: MutableMapping[Tuple[int, int], int] = dataclasses.field(
      default_factory=dict)


def note_event_data_to_events(
  state: Optional[NoteEncodingState],
  value: NoteEventData,
  codec: event_codec.Codec,
) -> Sequence[event_codec.Event]:
  """Convert note event data to a sequence of events."""
  # NoteEventData(pitch=24, velocity=93, program=0, is_drum=False, instrument=None)

  if value.velocity is None:
    # onsets only, no program or velocity
    return (event_codec.Event('pitch', value.pitch), )

  num_velocity_bins = vocabularies.num_velocity_bins_from_codec(codec)
  velocity_bin = vocabularies.velocity_to_bin(value.velocity, num_velocity_bins)
  # 似乎比例系数为1，velocity不变

  if value.program is None:
    # onsets + offsets + velocities only, no programs
    if state is not None:
      state.active_pitches[(value.pitch, 0)] = velocity_bin
    return (event_codec.Event('velocity', velocity_bin), event_codec.Event('pitch', value.pitch), )

  if value.is_drum:
    # drum events use a separate vocabulary
    return (event_codec.Event('velocity', velocity_bin), event_codec.Event('drum', value.pitch), )

  # program + velocity + pitch
  if state is not None:
    state.active_pitches[(value.pitch, value.program)] = velocity_bin
  return (
    event_codec.Event('program', value.program),
    event_codec.Event('velocity', velocity_bin),
    event_codec.Event('pitch', value.pitch)
  )
  # TODO 实际上都是钢琴曲，第一个program都是1258 十分冗余


def note_encoding_state_to_events(
  state: NoteEncodingState
) -> Sequence[event_codec.Event]:
  """Output program and pitch events for active notes plus a final tie event."""
  events = []
  for pitch, program in sorted(
      state.active_pitches.keys(), key=lambda k: k[::-1]):
    if state.active_pitches[(pitch, program)]:
      events += [event_codec.Event('program', program),
                 event_codec.Event('pitch', pitch)]
  events.append(event_codec.Event('tie', 0))
  return events


@dataclasses.dataclass
class NoteDecodingState:
  """Decoding state for note transcription."""
  current_time: float = 0.0
  # velocity to apply to subsequent pitch events (zero for note-off)
  current_velocity: int = DEFAULT_VELOCITY
  # program to apply to subsequent pitch events
  current_program: int = 0
  # onset time and velocity for active pitches and programs: {(pitch, current_program): (onset time, velocity)}
  active_pitches: MutableMapping[Tuple[int, int], Tuple[float, int]] = dataclasses.field(default_factory=dict)
  # pitches (with programs) to continue from previous segment
  tied_pitches: MutableSet[Tuple[int, int]] = dataclasses.field(default_factory=set)
  # whether or not we are in the tie section at the beginning of a segment
  is_tie_section: bool = False
  # partially-decoded NoteSequence
  note_sequence: note_seq.NoteSequence = dataclasses.field(
    default_factory=lambda: note_seq.NoteSequence(ticks_per_quarter=220)
  )


def decode_note_onset_event(
  state: NoteDecodingState,
  time: float,
  event: event_codec.Event,
  codec: event_codec.Codec,
) -> None:
  """Process note onset event and update decoding state."""
  if event.type == 'pitch':
    state.note_sequence.notes.add(
        start_time=time, end_time=time + DEFAULT_NOTE_DURATION,
        pitch=event.value, velocity=DEFAULT_VELOCITY)
    state.note_sequence.total_time = max(state.note_sequence.total_time,
                                         time + DEFAULT_NOTE_DURATION)
  else:
    raise ValueError('unexpected event type: %s' % event.type)


def _add_note_to_sequence(
    ns: note_seq.NoteSequence,
    start_time: float, end_time: float, pitch: int, velocity: int,
    program: int = 0, is_drum: bool = False
) -> None:

  end_time = max(end_time, start_time + MIN_NOTE_DURATION)
  ns.notes.add(
    start_time=start_time, end_time=end_time,
    pitch=pitch, velocity=velocity, program=program, is_drum=is_drum
  )
  ns.total_time = max(ns.total_time, end_time)


def decode_note_event(
  state: NoteDecodingState,
  time: float,
  event: event_codec.Event,
  codec: event_codec.Codec
) -> None:
  """Process note event and update decoding state."""

  if time < state.current_time:
    raise ValueError(f'event time < current time, {time} < {state.current_time}')
  state.current_time = time

  if event.type == 'pitch':
    pitch = event.value
    active_key = (pitch, state.current_program)

    # "tied" pitch
    if state.is_tie_section:
      if active_key not in state.active_pitches:
        raise ValueError(f'inactive pitch/program in tie section: {pitch}/{state.current_program}')
      if active_key in state.tied_pitches:
        raise ValueError(f'pitch/program is already tied: {pitch}/{state.current_program}')
      state.tied_pitches.add(active_key)
    
    # note offset
    elif state.current_velocity == 0:
      if active_key not in state.active_pitches:
        raise ValueError(f'note-off for inactive pitch/program: {pitch}/{state.current_program}')
      onset_time, onset_velocity = state.active_pitches.pop(active_key)
      # 使用 active_pitches 记录未结束的音符，此时就能方便地匹配
      _add_note_to_sequence(
        state.note_sequence, start_time=onset_time, end_time=time,
        pitch=pitch, velocity=onset_velocity, program=state.current_program
      )

    # note onset
    else:
      if active_key in state.active_pitches:
        onset_time, onset_velocity = state.active_pitches.pop(active_key)
        _add_note_to_sequence(
          state.note_sequence, start_time=onset_time, end_time=time,
          pitch=pitch, velocity=onset_velocity, program=state.current_program
        )
        # 该音符已经处于激活状态（被弹奏且仍未结束），此时又有开始事件
        # 虽然不会发生这种情况，但还是处理一下：弹出原来的并记录。这样根据下方语句就会开始一个新的该音符事件
      state.active_pitches[active_key] = (time, state.current_velocity)

  # drum onset
  elif event.type == 'drum':
    if state.current_velocity == 0:
      raise ValueError('velocity cannot be zero for drum event')
      # 鼓没有结束事件
    offset_time = time + DEFAULT_NOTE_DURATION
    _add_note_to_sequence(
      state.note_sequence, start_time=time, end_time=offset_time,
      pitch=event.value, velocity=state.current_velocity, is_drum=True
    )

  # velocity change
  elif event.type == 'velocity':
    num_velocity_bins = vocabularies.num_velocity_bins_from_codec(codec)
    velocity = vocabularies.bin_to_velocity(event.value, num_velocity_bins)
    state.current_velocity = velocity
  
  # program change
  elif event.type == 'program':
    state.current_program = event.value
  
  # end of tie section; end active notes that weren't declared tied
  elif event.type == 'tie':
    if not state.is_tie_section:
      raise ValueError('tie section end event when not in tie section')
    for (pitch, program) in state.active_pitches.keys():
      if (key := (pitch, program)) not in state.tied_pitches:
        onset_time, onset_velocity = state.active_pitches.pop(key)
        _add_note_to_sequence(
          state.note_sequence, start_time=onset_time, end_time=state.current_time,
          pitch=pitch, velocity=onset_velocity, program=program
        )
    state.is_tie_section = False
  
  else:
    raise ValueError('unexpected event type: %s' % event.type)


def begin_tied_pitches_section(state: NoteDecodingState) -> None:
  """Begin the tied pitches section at the start of a segment."""
  state.tied_pitches = set()
  state.is_tie_section = True


def flush_note_decoding_state(
  state: NoteDecodingState
) -> note_seq.NoteSequence:
  """End all active notes and return resulting NoteSequence."""
  
  for onset_time, _ in state.active_pitches.values():
    state.current_time = max(state.current_time, onset_time + MIN_NOTE_DURATION)
    # 根据仍未结束的音符更新current_time

  for (pitch, program) in state.active_pitches.keys():
    onset_time, onset_velocity = state.active_pitches.get((pitch, program))
    _add_note_to_sequence(
      state.note_sequence, start_time=onset_time, end_time=state.current_time,
      pitch=pitch, velocity=onset_velocity, program=program
    )
  assign_instruments(state.note_sequence)
  return state.note_sequence


class NoteEncodingSpecType(run_length_encoding.EventEncodingSpec):
  pass


# encoding spec for modeling onsets and offsets
NoteEncodingSpec = NoteEncodingSpecType(
  init_encoding_state_fn=lambda: None,
  encode_event_fn=note_event_data_to_events,
  encoding_state_to_events_fn=None,
  init_decoding_state_fn=NoteDecodingState,
  begin_decoding_segment_fn=lambda state: None,
  decode_event_fn=decode_note_event,
  flush_decoding_state_fn=flush_note_decoding_state
)
