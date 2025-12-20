import random
import copy
from .note import Note
from .melody import Melody
from .constants import PITCH_VALUES, DURATIONS, TARGET_BEATS

class GeneticOperators:
    
    def __init__(self, mutation_rate=0.1, crossover_rate=0.7):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def select_parent(self, population, tournament_size=3):
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda m: m.fitness_score)

    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1)
        
        crossover_point = random.uniform(4, 12)
        
        child_notes = []
        current_beats = 0
        
        for note in parent1.notes:
            if current_beats >= crossover_point:
                break
            child_notes.append(copy.deepcopy(note))
            current_beats += note.duration
        
        p2_current = 0
        for note in parent2.notes:
            if current_beats >= TARGET_BEATS:
                break
            
            if p2_current < crossover_point:
                p2_current += note.duration
                continue
            
            remaining = TARGET_BEATS - current_beats
            if note.duration <= remaining:
                child_notes.append(copy.deepcopy(note))
                current_beats += note.duration
            else:
                adjusted_note = copy.deepcopy(note)
                adjusted_note.duration = remaining
                child_notes.append(adjusted_note)
                current_beats += remaining
        
        self._fix_total_beats(child_notes)
        return Melody(child_notes)

    def mutate(self, melody):
        mutated = copy.deepcopy(melody)
        
        for note in mutated.notes:
            if random.random() < self.mutation_rate:
                shift = random.randint(-2, 2)
                new_pitch = note.pitch + shift
                if new_pitch in PITCH_VALUES:
                    note.pitch = new_pitch
            
            if random.random() < self.mutation_rate:
                note.duration = random.choice(DURATIONS)
        
        # Re-normalize to exactly TARGET_BEATS
        # current_beats = sum(n.duration for n in mutated.notes)
        # if abs(current_beats - TARGET_BEATS) > 0.1:
        #     if mutated.notes:
        #         adjustment = TARGET_BEATS - current_beats
        #         mutated.notes[-1].duration = max(0.5, mutated.notes[-1].duration + adjustment)
        
        return mutated

    def _snap(self, duration: float) -> float:
        return max(0.5, round(duration * 2) / 2)

    def _fix_total_beats(self, notes):
        if not notes:
            return
        max_dur = max(DURATIONS)

        for n in notes:
            n.duration = self._snap(n.duration)

        total = sum(n.duration for n in notes)
        delta = TARGET_BEATS - total

        while delta > 1e-6:
            dur = self._snap(min(delta, max_dur))
            notes.append(Note(random.choice(PITCH_VALUES), dur))
            delta -= dur

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
            if need > 1e-6 and len(notes) > 1:
                notes.pop()
                self._fix_total_beats(notes)
