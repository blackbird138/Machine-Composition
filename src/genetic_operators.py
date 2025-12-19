"""
Genetic operators: crossover, mutation, and selection.
"""

import random
import copy
from .note import Note
from .melody import Melody
from .constants import PITCH_VALUES, DURATIONS, TARGET_BEATS


class GeneticOperators:
    """Genetic algorithm operators for melody evolution."""
    
    def __init__(self, mutation_rate=0.1, crossover_rate=0.7):
        """
        Initialize genetic operators.
        
        Args:
            mutation_rate (float): Probability of mutation
            crossover_rate (float): Probability of crossover
        """
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def select_parent(self, population, tournament_size=3):
        """
        Tournament selection: pick best from random subset.
        
        Args:
            population (list): List of Melody objects
            tournament_size (int): Number of individuals in tournament
            
        Returns:
            Melody: Selected parent
        """
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda m: m.fitness_score)

    def crossover(self, parent1, parent2):
        """
        Single-point crossover: combine two parent melodies.
        
        Args:
            parent1 (Melody): First parent
            parent2 (Melody): Second parent
            
        Returns:
            Melody: Child melody
        """
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1)
        
        # Find crossover point (in beats)
        crossover_point = random.uniform(4, 12)  # Middle measures
        
        child_notes = []
        current_beats = 0
        
        # Take from parent1 until crossover point
        for note in parent1.notes:
            if current_beats >= crossover_point:
                break
            child_notes.append(copy.deepcopy(note))
            current_beats += note.duration
        
        # Fill rest from parent2
        p2_current = 0
        for note in parent2.notes:
            if current_beats >= TARGET_BEATS:
                break
            
            # Skip notes from parent2 until we pass crossover point
            if p2_current < crossover_point:
                p2_current += note.duration
                continue
            
            remaining = TARGET_BEATS - current_beats
            if note.duration <= remaining:
                child_notes.append(copy.deepcopy(note))
                current_beats += note.duration
            else:
                # Adjust last note to fit exactly
                adjusted_note = copy.deepcopy(note)
                adjusted_note.duration = remaining
                child_notes.append(adjusted_note)
                current_beats += remaining
        
        self._fix_total_beats(child_notes)
        return Melody(child_notes)

    def mutate(self, melody):
        """
        Modify melody by changing pitches or durations.
        
        Args:
            melody (Melody): Melody to mutate
            
        Returns:
            Melody: Mutated melody
        """
        mutated = copy.deepcopy(melody)
        
        for note in mutated.notes:
            # Pitch mutation
            if random.random() < self.mutation_rate:
                shift = random.randint(-2, 2)
                new_pitch = note.pitch + shift
                if new_pitch in PITCH_VALUES:
                    note.pitch = new_pitch
            
            # Duration mutation
            if random.random() < self.mutation_rate:
                note.duration = random.choice(DURATIONS)
        
        # Re-normalize to exactly TARGET_BEATS
        # current_beats = sum(n.duration for n in mutated.notes)
        # if abs(current_beats - TARGET_BEATS) > 0.1:
        #     if mutated.notes:
        #         adjustment = TARGET_BEATS - current_beats
        #         mutated.notes[-1].duration = max(0.5, mutated.notes[-1].duration + adjustment)
        
        return mutated

    # -----------------------
    # Helpers
    # -----------------------
    def _snap(self, duration: float) -> float:
        """Snap duration to nearest 0.5 beat (eighth-note base)."""
        return max(0.5, round(duration * 2) / 2)

    def _fix_total_beats(self, notes):
        """Ensure the melody totals TARGET_BEATS with 0.5-beat granularity."""
        if not notes:
            return
        max_dur = max(DURATIONS)

        # Snap all durations to 0.5 grid
        for n in notes:
            n.duration = self._snap(n.duration)

        total = sum(n.duration for n in notes)
        delta = TARGET_BEATS - total

        # If we are short, append filler notes (do not inflate existing notes beyond max_dur)
        while delta > 1e-6:
            dur = self._snap(min(delta, max_dur))
            notes.append(Note(random.choice(PITCH_VALUES), dur))
            delta -= dur

        # If we are long, trim from the tail without going below 0.5
        if delta < -1e-6:
            need = -delta
            for i in range(len(notes) - 1, -1, -1):
                reducible = max(0.0, notes[i].duration - 0.5)
                take = min(reducible, need)
                if take > 0:
                    notes[i].duration = self._snap(notes[i].duration - take)
                    need -= take
                if need <= 1e-6:
                    break
            # If still too long, drop the last note if it exists and retry once
            if need > 1e-6 and len(notes) > 1:
                notes.pop()
                self._fix_total_beats(notes)
