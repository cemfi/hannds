"""Tests for kalman_mapper.py"""

from kalman_mapper import MidiEvent, KalmanMapper, HandConstraints
import kalman_mapper


def test_constraints():
    constraints = HandConstraints()
    event = MidiEvent(100, is_note_on=True, when=0.0, is_left=False)
    constraints.midi_event(event)
    event = MidiEvent(20, is_note_on=True, when=0.0, is_left=True)
    constraints.midi_event(event)
    assert constraints.right_hand() == [100]
    assert constraints.left_hand() == [20]
    event = MidiEvent(100, is_note_on=False, when=1.0, is_left=False)
    constraints.midi_event(event)
    assert constraints.right_hand() == []
    assert constraints.left_hand() == []


def test_kalman_mapper():
    mapper = KalmanMapper()
    event = MidiEvent(100, is_note_on=True, when=0.0, is_left=False)
    mapper.midi_event(event)
    assert mapper.last_was_left_hand == event.is_left
    event = MidiEvent(20, is_note_on=True, when=0.0, is_left=True)
    mapper.midi_event(event)
    assert mapper.last_was_left_hand == event.is_left

    event = MidiEvent(20, is_note_on=False, when=1.0, is_left=True)
    mapper.midi_event(event)
    event = MidiEvent(100, is_note_on=False, when=1.0, is_left=False)
    mapper.midi_event(event)

    event = MidiEvent(30, is_note_on=True, when=1.0, is_left=True)
    mapper.midi_event(event)
    assert mapper.last_was_left_hand == event.is_left


def test_kalman_main():
    kalman_mapper.main(debug=True)
