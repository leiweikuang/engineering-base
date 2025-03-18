# This script solves the Knapsack problem using brute force and genetic algorithm
from collections import namedtuple
import random
from typing import Callable
from functools import partial

# Define all types required in this problem

# Genome is a list of integers, 0 means not bringing the item, 1 means bringing the item
Genome = list[int]
# The population is a list of genomes
Population = list[Genome]

# Things is a namedtuple of the weight and the value of the thing
Thing = namedtuple("Thing", ["name", "weight", "value"])

# Define functions as callables
# A generic fitness function that takes a genome and returns a fitness value
FitnessFunc = Callable[[Genome], int]

# Define functions as callables
# The Populate function that takes nothing and generates the population
PopulateFunc = Callable[[], Population]

# The Selection function that takes the population and the fitness function, and returns the selected 2 parent genomes
SelectionFunc = Callable[[Population, FitnessFunc], tuple[Genome, Genome]]

# Crossover function that takes 2 parent genomes and returns 2 child genomes
CrossoverFunc = Callable[[Genome, Genome], tuple[Genome, Genome]]

# Mutation function that takes a genome and a mutation rate, and returns the mutated genome
MutationRate = float
MutationFunc = Callable[[Genome, MutationRate], Genome]

# Define the list of things to be selected
things: list[Thing] = [
    Thing("laptop", 500, 2200),
    Thing("headphones", 150, 160),
    Thing("coffee mug", 60, 350),
    Thing("notepad", 40, 333),
    Thing("water bottle", 30, 192),
]

# Define more things to check bigger population
more_things: list[Thing] = [
    Thing("mints", 5, 25),
    Thing("socks", 10, 38),
    Thing("tissues", 15, 80),
    Thing("phone", 500, 200),
    Thing("baseball cap", 100, 70),
]


def generate_genome(length: int) -> Genome:
    # Generates a random genome of length length
    genome = [random.randint(0, 1) for _ in range(length)]
    return genome


def generate_population(population_size: int, genome_length: int) -> Population:
    # This function generates the population using the generate_genome function
    # until the population_size is reached
    population = []
    for _ in range(population_size):

        population.append(generate_genome(genome_length))

    return population


def fitness_function(genome: Genome, things: list[Thing], weight_limit: int) -> int:
    """This function calculates the fitness of the genome

    Args:
        genome (Genome): A genome of which the fitness is to be calculated
        things (list[Thing]): a list of things that are packed
        weight_limit(int): the maximum weight that can be brought in grams

    Returns:
        Fitness: fitness value
    """

    # Check if the genome is valid
    if len(genome) != len(things):
        raise ValueError("Genome and things must have the same length")

    # Initiate the weight and value of the genome
    weight = 0
    value = 0

    # Iterate through the genome and calculate the weight and value
    for i in range(len(things)):
        # if genome at this index is 1, add the weight and value of the thing
        if genome[i] == 1:
            # index 0 is the weight, index 1 is the value
            weight += things[i].weight
            value += things[i].value
        # if it's not 1, then it's a 0, so we don't bring it

        # Check if the weight is greater than the weight limit
        if weight > weight_limit:
            # If it is, immediate return fitness of 0
            return 0

    return value


def selection_pair(
    population: Population, fitness_function: FitnessFunc
) -> tuple[Genome, Genome]:
    """This function selects two genomes from the population based on their fitness
    for reproduction.

    Args:
        population (Population): _description_
        fitness_function (Fitness): _description_

    Returns:
        tuple[Genome, Genome]: _description_
    """

    # Use the list of fitness values as weighting for random choices, to increase
    # the probability of higher fitness genomes being selected
    # TODO: Check if a genome contains all required input to fitness_function
    weights = [fitness_function(genome) for genome in population]

    # We draw 2 parents from the population
    selected_genome = random.choices(population=population, weights=weights, k=2)

    return selected_genome


def single_point_crossover(parent1: Genome, parent2: Genome) -> tuple[Genome, Genome]:
    """This function takes two genomes and performs single point crossover

    Args:
        parent1 (Genome): Genome 1
        parent2 (Genome): Genome 2

    Returns:
        Child Genomes (Tuple[Genome, Genome]): Child Genomes
    """
    # Check that the genomes are the same length
    if len(parent1) != len(parent2):
        raise ValueError("Parents must be the same length")

    # Check genome length is greater than 2
    if len(parent1) < 2:
        return (parent1, parent2)

    # define random index
    index = random.randint(0, len(parent1) - 1)

    # create a new genome with the crossover point
    child1 = parent1[:index] + parent2[index:]
    child2 = parent2[:index] + parent1[index:]

    return (child1, child2)


def mutation(genome: Genome, mutation_rate: float) -> Genome:
    """_summary_

    Args:
        genome (Genome): Genome to be mutated
        mutation_rate (float): mutation rate, between 0 and 1

    Returns:
        Genome: mutated genome
    """

    # select random index to mutate
    index = random.randint(0, len(genome) - 1)

    # At that index, check if random is greater than mutation rate
    # if random is less than mutation rate, mutate the genome
    if random.random() < mutation_rate:
        # If it is, mutate the genome
        genome[index] = abs(1 - genome[index])

    return genome


# Define the function that runs the evolution
def run_evolution(
    populate_func: PopulateFunc,
    fitness_func: FitnessFunc = fitness_function,
    selection_func: SelectionFunc = selection_pair,
    crossover_func: CrossoverFunc = single_point_crossover,
    mutation_func: MutationFunc = mutation,
    mutation_rate: MutationRate = 0.5,
    max_generations: int = 100,
    fitness_limit: int = 100,
) -> tuple[Population, int, int]:
    # Create first generation
    population = populate_func()

    # Create for loop over max_generations
    for i in range(max_generations):
        # Sort population by fitness from highest to lowest
        # population = sorted(population, key=fitness_func, reverse=True)
        population = sorted(
            population, key=lambda genome: fitness_func(genome), reverse=True
        )

        # Check if fitness limit is reached
        if fitness_func(population[0]) >= fitness_limit:
            # If it is, return the population and the generation number
            return (population, i, fitness_func(population[0]))

        # Include elitism, always preseve the top 2 genomes
        next_population = population[0:2]

        # Select parent pairs based on fitness. Note that we've already reserved the top 2 genomes
        for j in range(int((len(population) - 2) / 2)):
            # Note that we can have overlapping parents, we just need to make sure the
            # next generation is the same length
            parents = selection_func(population, fitness_func)

            # Generate child genomes
            child1, child2 = crossover_func(parents[0], parents[1])

            # Mutate child genomes
            child1 = mutation_func(child1, mutation_rate)
            child2 = mutation_func(child2, mutation_rate)

            # Add child genomes to next population
            next_population.append(child1)
            next_population.append(child2)

        # Check to make sure the next population is the same length as the current population
        if len(next_population) != len(population):
            raise ValueError(
                "Next population is not the same length as the current population"
            )

        # Print current generation result
        print(
            f"""Generation {i},
        value: {fitness_func(population[0])},
        weight: {genome_to_things(population[0])[1]},
        things:{genome_to_things(population[0])[0]}"""
        )

        # Replace the current population with the next population
        population = next_population

    # After all is done, return the population and the generation number
    # sort it one last time in case we never reached the fitness limit
    population = sorted(population, key=fitness_func, reverse=True)

    return population, i, fitness_func(population[0])


def genome_to_things(genome: Genome) -> tuple[list, int]:
    result = []
    weight = []
    for i in range(len(genome)):
        if genome[i] == 1:
            result.append(things[i].name)
            weight.append(things[i].weight)
    total_weight = sum(weight)
    return result, total_weight


# run the evolution function
# NOTE: use partial to predefine missing arguments in signature
if __name__ == "__main__":
    last_population, generation, fitness = run_evolution(
        populate_func=partial(
            generate_population, population_size=10, genome_length=len(things)
        ),
        fitness_func=partial(fitness_function, things=more_things, weight_limit=3000),
        selection_func=selection_pair,
        crossover_func=single_point_crossover,
        mutation_func=mutation,
        mutation_rate=0.9,
        max_generations=100,
        fitness_limit=2200,
    )
    print(
        f"""Generation {generation},
        value: {fitness},
        weight: {genome_to_things(last_population[0])[1]},
        things:{genome_to_things(last_population[0])[0]}"""
    )
