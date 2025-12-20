import copy
from .constants import PITCH_VALUES

class MusicalTransforms:
    
    @staticmethod
    def transpose(melody, semitones):
        transposed = copy.deepcopy(melody)
        for note in transposed.notes:
            new_pitch = note.pitch + semitones
            if new_pitch in PITCH_VALUES:
                note.pitch = new_pitch
        return transposed

    @staticmethod
    def inversion(melody, axis_pitch=67):
        inverted = copy.deepcopy(melody)
        for note in inverted.notes:
            interval = note.pitch - axis_pitch
            new_pitch = axis_pitch - interval
            if new_pitch in PITCH_VALUES:
                note.pitch = new_pitch
        return inverted

    @staticmethod
    def retrograde(melody):
        retrograded = copy.deepcopy(melody)
        retrograded.notes = list(reversed(retrograded.notes))
        return retrograded
