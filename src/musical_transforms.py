"""
Musical transformation operations: transposition, inversion, retrograde.
"""

import copy
from .constants import PITCH_VALUES


class MusicalTransforms:
    """Musical transformation operations for melodies."""
    
    @staticmethod
    def transpose(melody, semitones):
        """
        Transpose melody by given semitones.
        
        Args:
            melody (Melody): Melody to transpose
            semitones (int): Number of semitones to shift
            
        Returns:
            Melody: Transposed melody
        """
        transposed = copy.deepcopy(melody)
        for note in transposed.notes:
            new_pitch = note.pitch + semitones
            if new_pitch in PITCH_VALUES:
                note.pitch = new_pitch
        return transposed

    @staticmethod
    def inversion(melody, axis_pitch=67):
        """
        Invert melody around an axis pitch.
        
        Args:
            melody (Melody): Melody to invert
            axis_pitch (int): MIDI pitch to invert around (default: G4=67)
            
        Returns:
            Melody: Inverted melody
        """
        inverted = copy.deepcopy(melody)
        for note in inverted.notes:
            interval = note.pitch - axis_pitch
            new_pitch = axis_pitch - interval
            if new_pitch in PITCH_VALUES:
                note.pitch = new_pitch
        return inverted

    @staticmethod
    def retrograde(melody):
        """
        Reverse the melody (play backwards).
        
        Args:
            melody (Melody): Melody to reverse
            
        Returns:
            Melody: Retrograded melody
        """
        retrograded = copy.deepcopy(melody)
        retrograded.notes = list(reversed(retrograded.notes))
        return retrograded
