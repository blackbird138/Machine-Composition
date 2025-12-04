"""
Population initialization for genetic algorithm.
"""

import random
from .note import Note
from .melody import Melody
from .constants import PITCH_VALUES, DURATIONS, TARGET_BEATS


class PopulationGenerator:
    """Generates initial population of random melodies."""
    
    @staticmethod
    def generate(population_size):
        """
        Generate initial population of random melodies.
        
        Args:
            population_size (int): Number of melodies to generate
            
        Returns:
            list: List of Melody objects
        """
        population = []
        
        for _ in range(population_size):
            melody = PopulationGenerator._create_random_melody()
            population.append(melody)
        
        print(f"Generated initial population of {len(population)} melodies.")
        return population

    @staticmethod
    def _create_random_melody():
        """
        Create a single random melody.
        
        Returns:
            Melody: Random melody with TARGET_BEATS duration
        """
        melody = Melody()
        current_beats = 0
        
        while current_beats < TARGET_BEATS:
            pitch = random.choice(PITCH_VALUES)
            
            # Ensure we don't exceed target beats
            remaining = TARGET_BEATS - current_beats
            possible_durations = [d for d in DURATIONS if d <= remaining]
            
            if not possible_durations:
                duration = 0.5  # Fallback to eighth note
            else:
                duration = random.choice(possible_durations)
            
            melody.add_note(Note(pitch, duration))
            current_beats += duration
        
        return melody
