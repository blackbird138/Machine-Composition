"""
Main entry point for the music composition genetic algorithm.
"""

from src.genetic_algorithm import GeneticAlgorithm
from src.exporter import MelodyExporter
from src.fitness import FitnessEvaluator


def main():
    """Run the genetic algorithm and export results."""
    print("=" * 60)
    print("Genetic Algorithm for Music Composition")
    print("=" * 60)
    
    # Initialize genetic algorithm
    ga = GeneticAlgorithm(
<<<<<<< Updated upstream
        population_size=200,
=======
        population_size=100,
>>>>>>> Stashed changes
        mutation_rate=0.15,
        crossover_rate=0.7
    )
    
    # Generate initial population
    ga.generate_initial_population()
    
    # Show a sample from initial population
    print("\nSample from initial population:")
    print(f"Melody 0: {ga.population[0].to_abc()}")
    
    # Evolve for 100 generations
    best_melodies = ga.evolve(generations=100)
    
    # Display and export results
    print("\n" + "=" * 60)
    print("Top 3 Melodies:")
    print("=" * 60)
    
    exporter = MelodyExporter()
<<<<<<< Updated upstream

    for i, melody in enumerate(best_melodies[:10]):

=======
    
    for i, melody in enumerate(best_melodies[:10]):
>>>>>>> Stashed changes
        print(f"\n--- Melody {i+1} (Fitness: {melody.fitness_score:.2f}) ---")
        print(f"ABC: {melody.to_abc()}")
        print(f"Notes: {melody.notes}")
        
        # Export to files
        exporter.to_abc_file(melody, f"melody_{i+1}.abc", f"Generated Melody {i+1}")
        exporter.to_midi_file(melody, f"melody_{i+1}.mid")
    
    print("\n" + "=" * 60)
    print("Complete! Check the generated ABC and MIDI files.")
    print("Visit https://abcjs.net/abcjs-editor.html to play ABC notation")
    print("=" * 60)


if __name__ == "__main__":
    main()
