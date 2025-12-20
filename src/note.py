from .constants import MIDI_TO_PITCH

class Note:
    
    def __init__(self, pitch, duration):
        self.pitch = pitch
        self.duration = duration

    def __repr__(self):
        return f"Note({MIDI_TO_PITCH.get(self.pitch, self.pitch)}, {self.duration})"

    def to_abc(self):
        name = MIDI_TO_PITCH.get(self.pitch, '')
        if not name:
            return ""
        
        accidental = ""
        if '#' in name:
            accidental = "^"
            base_name = name.replace('#', '')
        else:
            base_name = name
        
        note_char = base_name[0]
        octave = int(base_name[-1])
        
        abc_str = accidental + note_char
        
        if octave == 3:
            abc_str += ","
        elif octave == 5:
            abc_str = abc_str.lower()
        elif octave >= 6:
            abc_str = abc_str.lower() + "'" * (octave - 5)

        units = int(self.duration * 2)
        if units > 1:
            abc_str += str(units)
            
        return abc_str
