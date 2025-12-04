"""
Note class representing a musical note with pitch and duration.
"""

from .constants import MIDI_TO_PITCH


class Note:
    """Represents a single musical note."""
    
    def __init__(self, pitch, duration):
        """
        Initialize a note.
        
        Args:
            pitch (int): MIDI pitch value (53-79 for F3-G5)
            duration (float): Duration in beats (quarter notes)
        """
        self.pitch = pitch
        self.duration = duration

    def __repr__(self):
        """String representation of the note."""
        return f"Note({MIDI_TO_PITCH.get(self.pitch, self.pitch)}, {self.duration})"

    def to_abc(self):
        """
        Convert note to ABC notation string.
        
        Returns:
            str: ABC notation representation
        """
        name = MIDI_TO_PITCH.get(self.pitch, '')
        if not name:
            return ""
        
        # Handle accidentals (sharps)
        accidental = ""
        if '#' in name:
            accidental = "^"
            base_name = name.replace('#', '')
        else:
            base_name = name
            
        # Extract note character and octave
        note_char = base_name[0]
        octave = int(base_name[-1])
        
        # Build ABC string with octave notation
        # C4 is middle C (uppercase), C5 is lowercase, C3 needs comma
        abc_str = accidental + note_char
        
        if octave == 3:
            abc_str += ","
        elif octave == 5:
            abc_str = abc_str.lower()
        elif octave >= 6:
            abc_str = abc_str.lower() + "'" * (octave - 5)
        # octave 4 is default (uppercase)
        
        # Handle duration
        # Base length L:1/8 means:
        # 0.5 beat = 1 unit (eighth note) - no suffix
        # 1.0 beat = 2 units (quarter note) - add "2"
        # 2.0 beat = 4 units (half note) - add "4"
        units = int(self.duration * 2)
        if units > 1:
            abc_str += str(units)
            
        return abc_str
