class Melody:
    
    def __init__(self, notes=None):
        self.notes = notes if notes else []
        self.fitness_score = 0.0

    def add_note(self, note):
        self.notes.append(note)

    def length_beats(self):
        return sum(n.duration for n in self.notes)
    
    def to_abc(self):
        tokens = []
        measure_beats = 0.0
        accidental_state = {}

        for note in self.notes:
            remaining = note.duration
            while remaining > 1e-9:
                if abs(measure_beats - 4.0) < 1e-6:
                    tokens.append("|")
                    measure_beats = 0.0
                    accidental_state = {}

                remaining_in_bar = 4.0 - measure_beats
                part = min(remaining, remaining_in_bar)

                token = self._note_to_abc(note, accidental_state, duration_override=part)
                if remaining > part + 1e-9:
                    token += "-"

                tokens.append(token)
                measure_beats += part
                remaining -= part

            if abs(measure_beats - 4.0) < 1e-6:
                tokens.append("|")
                measure_beats = 0.0
                accidental_state = {}

        if tokens and tokens[-1] == "|":
            tokens.pop()

        return " ".join(tokens)

    def _note_to_abc(self, note, accidental_state, duration_override=None):
        from .constants import MIDI_TO_PITCH

        duration = duration_override if duration_override is not None else note.duration

        name = MIDI_TO_PITCH.get(note.pitch, '')
        if not name:
            return ""

        is_sharp = '#' in name
        base_name = name.replace('#', '')
        note_char = base_name[0]
        octave = int(base_name[-1])

        key = (note_char, octave)

        prefix = ""
        if is_sharp:
            prefix = "^"
            accidental_state[key] = "sharp"
        else:
            if accidental_state.get(key) == "sharp":
                prefix = "="
            accidental_state[key] = "natural"

        abc_pitch = prefix + note_char
        if octave == 3:
            abc_pitch += ","
        elif octave == 5:
            abc_pitch = abc_pitch.lower()
        elif octave >= 6:
            abc_pitch = abc_pitch.lower() + "'" * (octave - 5)

        units = int(duration * 2 + 1e-6)
        if units > 1:
            abc_pitch += str(units)

        return abc_pitch

    def __repr__(self):
        return f"Melody(len={len(self.notes)}, score={self.fitness_score:.2f})"
