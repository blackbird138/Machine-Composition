"""
Main genetic algorithm implementation for melody evolution.
"""

import random
import copy
from .population import PopulationGenerator
from .fitness import FitnessEvaluator
from .genetic_operators import GeneticOperators
from .musical_transforms import MusicalTransforms


class GeneticAlgorithm:
    """Genetic algorithm for evolving melodies."""
    
    def __init__(self, population_size=20, mutation_rate=0.1, crossover_rate=0.7):
        """
        Initialize genetic algorithm.
        
        Args:
            population_size (int): Number of individuals in population
            mutation_rate (float): Probability of mutation
            crossover_rate (float): Probability of crossover
        """
        self.population_size = population_size
        self.population = []
        self.generation = 0
        
        # Initialize components
        self.fitness_evaluator = FitnessEvaluator()
        self.genetic_ops = GeneticOperators(mutation_rate, crossover_rate)
        self.transforms = MusicalTransforms()

    def generate_initial_population(self):
        """Generate initial population of random melodies."""
        self.population = PopulationGenerator.generate(self.population_size)

    def evolve(self, generations=50):
        """
        Run the genetic algorithm for specified generations.
        
        Args:
            generations (int): Number of generations to evolve
            
        Returns:
            list: Final population sorted by fitness (best first)
        """
        print(f"\nStarting evolution for {generations} generations...\n")
        
        # Evaluate initial population
        for melody in self.population:
            self.fitness_evaluator.evaluate(melody)
        
        # Evolution loop
        for gen in range(generations):
            self.generation = gen
            self.population = self._create_next_generation()
            
            # Print progress
            if gen % 10 == 0 or gen == generations - 1:
                self._print_generation_stats(gen)
        
        print("\nEvolution complete!")
        return sorted(self.population, key=lambda m: m.fitness_score, reverse=True)

    def _create_next_generation(self):
        """
        Create next generation using selection, crossover, and mutation.
        
        Returns:
            list: New population
        """
        new_population = []
        
        # Elitism: keep best 2 melodies
        sorted_pop = sorted(self.population, key=lambda m: m.fitness_score, reverse=True)
        new_population.extend([copy.deepcopy(sorted_pop[0]), copy.deepcopy(sorted_pop[1])])
        
        # Generate rest of population
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.genetic_ops.select_parent(self.population)
            parent2 = self.genetic_ops.select_parent(self.population)
            
            # Crossover
            child = self.genetic_ops.crossover(parent1, parent2)
            
            # Mutation
            child = self.genetic_ops.mutate(child)
            
            # Occasionally apply musical transforms
            if random.random() < 0.1:
                child = self._apply_random_transform(child)
            
            # Evaluate fitness
            self.fitness_evaluator.evaluate(child)
            new_population.append(child)
        
        return new_population

    def _apply_random_transform(self, melody):
        """
        Apply a random musical transformation.
        
        Args:
            melody (Melody): Melody to transform
            
        Returns:
            Melody: Transformed melody
        """
        transform = random.choice(['transpose', 'inversion', 'retrograde'])
        
        if transform == 'transpose':
            return self.transforms.transpose(melody, random.choice([-2, -1, 1, 2]))
        elif transform == 'inversion':
            return self.transforms.inversion(melody)
        else:
            return self.transforms.retrograde(melody)

    def _print_generation_stats(self, generation):
        """Print statistics for current generation."""
        best = max(self.population, key=lambda m: m.fitness_score)
        avg_fitness = sum(m.fitness_score for m in self.population) / len(self.population)
        print(f"Gen {generation}: Best fitness = {best.fitness_score:.2f}, "
              f"Avg fitness = {avg_fitness:.2f}")
