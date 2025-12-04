"""
Fitness evaluation functions for melody quality assessment.
"""

from .constants import C_MAJOR_SCALE, DURATIONS


class FitnessEvaluator:
    """Evaluates melody quality based on musical heuristics."""
    
    def __init__(self, scale=None):
        """
        Initialize fitness evaluator.
        
        Args:
            scale (list): List of MIDI pitch values for scale adherence check
        """
        self.scale = scale if scale else C_MAJOR_SCALE

    def evaluate(self, melody):
        """
        Evaluate melody fitness based on multiple criteria.
        
        Args:
            melody (Melody): Melody to evaluate
            
        Returns:
            float: Fitness score (0-100)
        """
        notes = melody.notes
        if not notes:
            return 0.0
        
        # Calculate individual components
        scale_score = self._scale_adherence(notes)
        interval_score = self._melodic_contour(notes)
        rhythmic_score = self._rhythmic_variety(notes)
        
        # Total fitness
        total_score = scale_score + interval_score + rhythmic_score
        melody.fitness_score = total_score
        
        return total_score

    def _scale_adherence(self, notes):
        """
        Evaluate scale adherence (0-40 points).
        Prefer notes in the target scale.
        
        Args:
            notes (list): List of Note objects
            
        Returns:
            float: Scale adherence score
        """
        scale_notes = sum(1 for note in notes if note.pitch in self.scale)
        return (scale_notes / len(notes)) * 40

    def _melodic_contour(self, notes):
        """
        Evaluate melodic contour (0-30 points).
        Penalize large interval jumps.
        
        Args:
            notes (list): List of Note objects
            
        Returns:
            float: Melodic contour score
        """
        score = 30
        
        for i in range(len(notes) - 1):
            interval = abs(notes[i+1].pitch - notes[i].pitch)
            
            if interval > 12:  # More than an octave
                score -= 5
            elif interval > 7:  # More than a perfect fifth
                score -= 2
        
        return max(0, score)

    def _rhythmic_variety(self, notes):
        """
        Evaluate rhythmic variety (0-30 points).
        Reward different durations.
        
        Args:
            notes (list): List of Note objects
            
        Returns:
            float: Rhythmic variety score
        """
        durations = [n.duration for n in notes]
        unique_durations = len(set(durations))
        return (unique_durations / len(DURATIONS)) * 30
