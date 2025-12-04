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
        
        Returns:
            str: Space-separated ABC notation of all notes
        """
        return " ".join(n.to_abc() for n in self.notes)

    def __repr__(self):
        """String representation of the melody."""
        return f"Melody(len={len(self.notes)}, score={self.fitness_score:.2f})"
