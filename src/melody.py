"""
Melody class representing a sequence of notes.
"""


class Melody:
    """Represents a sequence of musical notes."""
    
    def __init__(self, notes=None):
        """
        Initialize a melody.
        
        Args:
            notes (list): List of Note objects
        """
        self.notes = notes if notes else []
        self.fitness_score = 0.0

    def add_note(self, note):
        """Add a note to the melody."""
        self.notes.append(note)

    def length_beats(self):
        """Calculate total duration in beats."""
        return sum(n.duration for n in self.notes)
    
    def to_abc(self):
        """
        Convert melody to ABC notation string.
        Ensures bar lines every 4 beats and ties notes that cross barlines.

        Returns:
            str: ABC notation of the melody with barlines.
        """
        tokens = []
        measure_beats = 0.0
        accidental_state = {}  # {(note_char, octave): 'sharp'|'natural'} for the current bar

        for note in self.notes:
            remaining = note.duration
            while remaining > 1e-9:
                # If a bar is complete, start a new one
                if abs(measure_beats - 4.0) < 1e-6:
                    tokens.append("|")
                    measure_beats = 0.0
                    accidental_state = {}

                remaining_in_bar = 4.0 - measure_beats
                part = min(remaining, remaining_in_bar)

                token = self._note_to_abc(note, accidental_state, duration_override=part)
                # If the note continues across the bar, tie it
                if remaining > part + 1e-9:
                    token += "-"

                tokens.append(token)
                measure_beats += part
                remaining -= part

            # If the bar is exactly filled after this note, insert barline
            if abs(measure_beats - 4.0) < 1e-6:
                tokens.append("|")
                measure_beats = 0.0
                accidental_state = {}

        # Remove trailing bar if present
        if tokens and tokens[-1] == "|":
            tokens.pop()

        return " ".join(tokens)

    def _note_to_abc(self, note, accidental_state, duration_override=None):
        """Convert a note to ABC string while handling bar-scoped accidentals."""
        from .constants import MIDI_TO_PITCH  # Local import to avoid circular deps

        duration = duration_override if duration_override is not None else note.duration

        name = MIDI_TO_PITCH.get(note.pitch, '')
        if not name:
            return ""

        is_sharp = '#' in name
        base_name = name.replace('#', '')
        note_char = base_name[0]
        octave = int(base_name[-1])

        key = (note_char, octave)

        # Determine accidental prefix: '^' for sharp; '=' to cancel prior sharp in measure
        prefix = ""
        if is_sharp:
            prefix = "^"
            accidental_state[key] = "sharp"
        else:
            if accidental_state.get(key) == "sharp":
                prefix = "="
            accidental_state[key] = "natural"

        # Build pitch with octave marks
        abc_pitch = prefix + note_char
        if octave == 3:
            abc_pitch += ","
        elif octave == 5:
            abc_pitch = abc_pitch.lower()
        elif octave >= 6:
            abc_pitch = abc_pitch.lower() + "'" * (octave - 5)

        # Duration (L:1/8 base)
        units = int(duration * 2 + 1e-6)
        if units > 1:
            abc_pitch += str(units)

        return abc_pitch

    def __repr__(self):
        """String representation of the melody."""
        return f"Melody(len={len(self.notes)}, score={self.fitness_score:.2f})"
