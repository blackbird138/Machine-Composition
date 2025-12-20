import random
import copy
from .population import PopulationGenerator
from .fitness import FitnessEvaluator
from .genetic_operators import GeneticOperators
from .musical_transforms import MusicalTransforms

class GeneticAlgorithm:
    
    def __init__(self, population_size=20, mutation_rate=0.1, crossover_rate=0.7, fitness_evaluator=None):
        self.population_size = population_size
        self.population = []
        self.generation = 0

        self.fitness_evaluator = fitness_evaluator or FitnessEvaluator()
        self.genetic_ops = GeneticOperators(mutation_rate, crossover_rate)
        self.transforms = MusicalTransforms()

    def generate_initial_population(self):
        self.population = PopulationGenerator.generate(self.population_size)

    def evolve(self, generations=50):
        print(f"\nStarting evolution for {generations} generations...\n")
        
        for melody in self.population:
            self.fitness_evaluator.evaluate(melody)
        
        for gen in range(generations):
            self.generation = gen
            self.population = self._create_next_generation()
            
            if gen % 10 == 0 or gen == generations - 1:
                self._print_generation_stats(gen)
        
        print("\nEvolution complete!")
        return sorted(self.population, key=lambda m: m.fitness_score, reverse=True)

    def _create_next_generation(self):
        new_population = []
        
        sorted_pop = sorted(self.population, key=lambda m: m.fitness_score, reverse=True)
        new_population.extend([copy.deepcopy(sorted_pop[0]), copy.deepcopy(sorted_pop[1])])
        
        while len(new_population) < self.population_size:
            parent1 = self.genetic_ops.select_parent(self.population)
            parent2 = self.genetic_ops.select_parent(self.population)
            
            child = self.genetic_ops.crossover(parent1, parent2)
            
            child = self.genetic_ops.mutate(child)
            
            if random.random() < 0.1:
                child = self._apply_random_transform(child)
            
            self.fitness_evaluator.evaluate(child)
            new_population.append(child)
        
        return new_population

    def _apply_random_transform(self, melody):
        transform = random.choice(['transpose', 'inversion', 'retrograde'])
        
        if transform == 'transpose':
            return self.transforms.transpose(melody, random.choice([-2, -1, 1, 2]))
        elif transform == 'inversion':
            return self.transforms.inversion(melody)
        else:
            return self.transforms.retrograde(melody)

    def _print_generation_stats(self, generation):
        best = max(self.population, key=lambda m: m.fitness_score)
        avg_fitness = sum(m.fitness_score for m in self.population) / len(self.population)
        print(f"Gen {generation}: Best fitness = {best.fitness_score:.2f}, "
              f"Avg fitness = {avg_fitness:.2f}")
