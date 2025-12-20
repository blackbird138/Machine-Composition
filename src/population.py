import random
from .note import Note
from .melody import Melody
from .constants import PITCH_VALUES, DURATIONS, TARGET_BEATS

class PopulationGenerator:
    
    @staticmethod
    def generate(population_size):
        population = []
        
        for _ in range(population_size):
            melody = PopulationGenerator._create_random_melody()
            population.append(melody)
        
        print(f"Generated initial population of {len(population)} melodies.")
        return population

    @staticmethod
    def _create_random_melody():
        melody = Melody()
        current_beats = 0
        
        while current_beats < TARGET_BEATS:
            pitch = random.choice(PITCH_VALUES)
            
            remaining = TARGET_BEATS - current_beats
            possible_durations = [d for d in DURATIONS if d <= remaining]
            
            if not possible_durations:
                duration = 0.5
            else:
                duration = random.choice(possible_durations)
            
            melody.add_note(Note(pitch, duration))
            current_beats += duration
        
        return melody
