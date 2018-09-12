import os
from collections import namedtuple

import mido
from mido.midifiles.tracks import _to_abstime

import hannds_data


# Data

def kalman_mapper_data(path, debug=False):
    data_dict = {}
    midi_files = hannds_data.get_files_from_path(path, ['*.mid', '*.midi'])

    for idx, midi_file in enumerate(midi_files):
        midi = mido.MidiFile(midi_file)
        assert midi.ticks_per_beat == 480
        midi_data = midi.tracks[1], midi.tracks[0]
        key = os.path.basename(midi_file)
        data_dict[key] = midi_data
        if debug:
            break

    _check_left_right(data_dict)
    joined_dict = {key: _join_tracks(*value) for (key, value) in data_dict.items()}
    return joined_dict


def _check_left_right(data):
    for key in data.keys():
        name_left = data[key][0][0].name
        name_right = data[key][1][0].name
        if 'links' not in name_left.lower():
            raise Exception(f'{key}: {name_left} does not match "links"')
        if 'rechts' not in name_right.lower():
            raise Exception(f'{key}: {name_right} does not match "rechts"')


def _join_tracks(left_track, right_track):
    default_tempo = mido.bpm2tempo(120)
    default_ticks_per_beat = 480

    messages = []
    for msg in _to_abstime(left_track):
        is_note_on = (msg.type == 'note_on')
        is_note_off = (msg.type == 'note_off')
        if is_note_on or is_note_off:
            time = mido.tick2second(msg.time, default_ticks_per_beat, default_tempo)
            event = MidiEvent(pitch=msg.note, is_note_on=is_note_on, when=time, is_left=True)
            messages.append(event)

    for msg in _to_abstime(right_track):
        is_note_on = (msg.type == 'note_on')
        is_note_off = (msg.type == 'note_off')
        if is_note_on or is_note_off:
            time = mido.tick2second(msg.time, default_ticks_per_beat, default_tempo)
            event = MidiEvent(pitch=msg.note, is_note_on=is_note_on, when=time, is_left=False)
            messages.append(event)

    messages.sort(key=lambda msg: msg.when)
    return messages


# Algorithm

MidiEvent = namedtuple('MidiEvent', ['pitch', 'is_note_on', 'when', 'is_left'])


class HandConstraints(object):

    def __init__(self):
        self.sounding_notes = [False] * 128
        self.right_hand_notes = []
        self.left_hand_notes = []

    def right_hand(self):
        if self.right_hand_notes is None:
            self._assign_notes()
        return self.right_hand_notes

    def left_hand(self):
        if self.left_hand_notes is None:
            self._assign_notes()
        return self.left_hand_notes

    def midi_event(self, event):
        self.right_hand_notes = None
        self.left_hand_notes = None
        self.sounding_notes[event.pitch] = event.is_note_on

    def _assign_notes(self):
        comfortable_hand_span = 14  # 14 semitones = an ninth
        self.right_hand_notes = []
        self.left_hand_notes = []

        lowest = self._lowest_note()
        if lowest == 127:
            return

        highest = self._highest_note()
        if highest == 0:
            return

        for i in range(128):
            if self.sounding_notes[i]:
                if (i <= lowest + comfortable_hand_span) and (i < highest - comfortable_hand_span):
                    self.left_hand_notes.append(i)
                elif (i > lowest + comfortable_hand_span) and (i >= highest - comfortable_hand_span):
                    self.right_hand_notes.append(i)

    def _lowest_note(self):
        for i in range(128):
            if self.sounding_notes[i]:
                return i
        return 127

    def _highest_note(self):
        for i in reversed(range(128)):
            if self.sounding_notes[i]:
                return i
        return 0


class KalmanMapper(object):
    def __init__(self):
        self.left_hand_pos = 43.0  # mLeftHandPosition
        self.right_hand_pos = 77.0
        self.left_hand_variance = 1000.0
        self.right_hand_variance = 1000.0
        self.hand_constraints = HandConstraints()

        self.time_last_rh = None
        self.time_last_lh = None
        self.last_was_left_hand = False  # the result

    def midi_event(self, event):
        variance_per_second = 20.0
        midi_variance = 20.0

        self.hand_constraints.midi_event(event)
        if not event.is_note_on:
            return

        assign_left = False
        for p in self.hand_constraints.left_hand():
            if p == event.pitch:
                assign_left = True

        assign_right = False
        for p in self.hand_constraints.right_hand():
            if p == event.pitch:
                assign_right = True

        if not assign_left and not assign_right:
            delta_rh = abs(self.right_hand_pos - event.pitch)
            delta_lh = abs(self.left_hand_pos - event.pitch)
            assign_right = delta_lh > delta_rh
            assign_left = not assign_right
            self.last_was_left_hand = assign_left

        if assign_left:
            if self.time_last_lh is not None:
                delta = event.when - self.time_last_lh
                self.left_hand_variance += delta * variance_per_second

            self.left_hand_pos += self.left_hand_variance / (self.left_hand_variance + midi_variance) * (event.pitch - self.left_hand_pos)
            self.left_hand_variance -= self.left_hand_variance / (self.left_hand_variance + midi_variance) * self.left_hand_variance
            self.time_last_lh = event.when
            self.last_was_left_hand = True

        if assign_right:
            if self.time_last_rh:
                delta = event.when - self.time_last_rh
                self.right_hand_variance += delta * variance_per_second

            self.right_hand_pos += self.right_hand_variance / (self.right_hand_variance + midi_variance) * (event.pitch - self.right_hand_pos)
            self.right_hand_variance -= self.right_hand_variance / (self.right_hand_variance + midi_variance) * self.right_hand_variance
            self.time_last_rh = event.when
            self.last_was_left_hand = False


# Evaluation


def evaluate_all(data):
    total_correct = 0
    total_wrong = 0
    for key in data.keys():
        correct, wrong = evaluate_piece(data, key)
        total_correct += correct
        total_wrong += wrong

    acc = total_correct / (total_correct + total_wrong) * 100.0
    out_str = f'Total accuracy = {acc:.2f}% ({total_correct} / {total_correct + total_wrong})'
    print('-' * len(out_str))
    print(out_str)


def evaluate_piece(data, key):
    print(key)
    mapper = KalmanMapper()
    correct_notes = 0
    wrong_notes = 0
    for event in data[key]:
        mapper.midi_event(event)
        if event.is_note_on:
            if mapper.last_was_left_hand == event.is_left:
                correct_notes += 1
            else:
                wrong_notes += 1
    accuracy = correct_notes / (correct_notes + wrong_notes) * 100.0
    print(f'Accuracy = {accuracy:.1f}% ({correct_notes} / {correct_notes + wrong_notes})')
    print()
    return correct_notes, wrong_notes


def main():
    data = kalman_mapper_data('data/', debug=False)
    evaluate_all(data)


if __name__ == '__main__':
    main()