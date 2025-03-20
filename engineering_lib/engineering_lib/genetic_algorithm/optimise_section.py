# This script solves the Knapsack problem using brute force and genetic algorithm
import random
from typing import Callable
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

# Define all types required in this problem

# Genome is a list of integers, 0 means not bringing the item, 1 means bringing the item
Genome = list[float]
# A genome contains a list of 4 floats,
# H: Total height (mm)
# h: Internal height (mm)
# a: Web thickness (mm)
# b: Flange width (mm)

# The population is a list of genomes
Population = list[Genome]

# Things is a namedtuple of the weight and the value of the thing
# Thing = namedtuple("Thing", ["name", "weight", "value"])

# Define functions as callables
# A generic fitness function that takes a genome and returns a fitness value
FitnessFunc = Callable[[Genome], float]

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


def generate_genome(spread: float = 0.5) -> Genome:
    # Generates a random genome of length 4

    def generate_values():
        H = np.random.normal(1, spread) * 100.0
        h = np.random.normal(1, spread) * 50.0
        a = np.random.normal(1, spread) * 25.0
        b = np.random.normal(1, spread) * 500

        # Make sure numbers make sense
        if (H - h < 25) or (b < a) or (H * h * a * b <= 0) or (a < 25):
            return generate_values()
        else:
            return [H, h, a, b]

    return generate_values()


def calc_area(genome: Genome) -> float:
    H = genome[0]
    h = genome[1]
    a = genome[2]
    b = genome[3]

    return (H - h) * b + a * h


def generate_population(population_size: int) -> Population:
    # This function generates the population using the generate_genome function
    # until the population_size is reached
    population = []
    for _ in range(population_size):

        population.append(generate_genome())

    return population


def fitness_function(
    genome: Genome,
    area_limit: float = 20000,
    min_width=350,
    min_thickness=25,
    penalty=0.5,
) -> float:
    # The fitness value is the second moment of inertia of the genome

    # Check if the genome is valid
    if len(genome) != 4:
        raise ValueError("Genome and things must be 4")

    # Check if the area limit is exceeded, if so, assume fitness of 0
    if calc_area(genome) > area_limit:
        return 0
    # Restriction 3: cannot be too thin
    elif genome[2] < min_thickness:
        return 0
    # Must be bigger than min width
    elif genome[3] < min_width:
        return 0
    elif genome[0] - genome[1] < min_thickness:
        return 0
    else:
        return max(0, second_moment_of_inertia(genome))


def second_moment_of_inertia(genome: Genome) -> float:
    H = genome[0]
    h = genome[1]
    a = genome[2]
    b = genome[3]

    I_x = (a * h**3 / 12) + (b / 12) * (H**3 - h**3)

    return I_x


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


def single_point_crossover(
    parent1: Genome, parent2: Genome, spread: float = 0.1
) -> tuple[Genome, Genome]:
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

    # define random index
    index = random.randint(0, len(parent1) - 1)

    # create two new genomes based on parents.
    # use the average of the parents as the base
    H_base = (parent1[0] + parent2[0]) / 2
    h_base = (parent1[1] + parent2[1]) / 2
    a_base = (parent1[2] + parent2[2]) / 2
    b_base = (parent1[3] + parent2[3]) / 2

    child = [H_base, h_base, a_base, b_base]

    # When generating off spring, the childs vary a bit
    child1 = [
        H_base * np.random.normal(1, spread),
        h_base * np.random.normal(1, spread),
        a_base * np.random.normal(1, spread),
        b_base * np.random.normal(1, spread),
    ]
    child2 = [
        H_base * np.random.normal(1, spread),
        h_base * np.random.normal(1, spread),
        a_base * np.random.normal(1, spread),
        b_base * np.random.normal(1, spread),
    ]

    return child1, child2
    # return child, child


def mutation(genome: Genome, mutation_rate: float, spread: float = 0.5) -> Genome:
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
        # If it is, mutate the genome by introducing a bit more randomness
        genome[index] = genome[index] * np.random.normal(1, spread)

    return genome


def calc_population_value(population: Population, fitness_func: FitnessFunc) -> list:
    # Calculate the population value
    population_value = []
    for genome in population:
        population_value.append(fitness_func(genome))
    return population_value


def is_ascending(lst):
    if within_1_percent(lst):
        return True
    else:
        return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))


def within_1_percent(lst):
    if sum(lst) == 0:
        return True

    if not lst:  # Handle empty list case
        return False

    min_val = min(lst)
    max_val = max(lst)

    return (max_val - min_val) / max_val <= 0.01


# Define the function that runs the evolution
def run_evolution(
    populate_func: PopulateFunc,
    fitness_func: FitnessFunc = fitness_function,
    selection_func: SelectionFunc = selection_pair,
    crossover_func: CrossoverFunc = single_point_crossover,
    mutation_func: MutationFunc = mutation,
    mutation_rate: MutationRate = 0.2,
    max_generations: int = 100,
    fitness_limit: int = 500000000,
) -> tuple[Population, int, int, list]:
    # Create first generation
    population = populate_func()
    fitness_list = []

    # Create for loop over max_generations
    for i in range(max_generations):
        # Sort population by fitness from highest to lowest
        # population = sorted(population, key=fitness_func, reverse=True)
        population = sorted(
            population, key=lambda genome: fitness_func(genome), reverse=True
        )

        # Check sorting is correct, must be in descending order
        population_value_list = calc_population_value(population, fitness_func)

        if is_ascending(population_value_list) is True:
            if within_1_percent(population_value_list) is True:
                pass
            else:
                raise ValueError("Population is in ascending order, and not within 1%")

        # Check if fitness limit is reached
        if fitness_func(population[0]) >= fitness_limit:
            # If it is, return the population and the generation number
            return (population, i, fitness_func(population[0]), fitness_list)

        # Include elitism if needed, always preserve the top 2 genomes
        next_population = population[0:2]
        # next_population = []

        # Select parent pairs based on fitness. Note that we've already reserved the top 2 genomes
        for j in range(int((len(population)) / 2) - 1):
            # for j in range(int((len(population)) / 2) ):
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
            Ix: {round(calc_population_value(population, fitness_func)[0],2)},
            Depth: {round(population[0][0], 2)},
            area: {round(calc_area(population[0]), 2)},
            flange thickness: {round((population[0][0] - population[0][1]) / 2, 2)},
            web thickness: {round(population[0][2], 2)},
        """
        )

        # Replace the current population with the next population
        population = next_population

        fitness_list.append(fitness_func(population[0]))

    # After all is done, return the population and the generation number
    # sort it one last time in case we never reached the fitness limit
    population = sorted(population, key=fitness_func, reverse=True)

    return population, i, fitness_func(population[0]), fitness_list


def plot_fitness(fitness_list, ax):
    """
    Plots the fitness values over generations.

    Parameters:
    fitness_list - A list of fitness values, where each index represents a generation.
    """
    generations = list(range(1, len(fitness_list) + 1))

    ax.plot(
        generations, fitness_list, marker="o", linestyle="-", color="g", label="Fitness"
    )

    # Labels and title
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness Evolution")


def generate_section_plot(genome: Genome, ax):
    # Generate a plot
    H = genome[0]
    h = genome[1]
    a = genome[2]
    b = genome[3]

    flange_thickness = (H - h) / 2  # Thickness of the top and bottom flanges

    # Define coordinates for the I-section
    x_coords = [
        -b / 2,
        b / 2,
        b / 2,
        a / 2,
        a / 2,
        b / 2,
        b / 2,
        -b / 2,
        -b / 2,
        -a / 2,
        -a / 2,
        -b / 2,
        -b / 2,
    ]
    y_coords = [
        H / 2,
        H / 2,
        H / 2 - flange_thickness,
        H / 2 - flange_thickness,
        -H / 2 + flange_thickness,
        -H / 2 + flange_thickness,
        -H / 2,
        -H / 2,
        -H / 2 + flange_thickness,
        -H / 2 + flange_thickness,
        H / 2 - flange_thickness,
        H / 2 - flange_thickness,
        H / 2,
    ]

    # Plot the section
    ax.plot(x_coords, y_coords, "k", linewidth=2)
    ax.set_xlim(-b, b)
    ax.set_ylim(-H / 2, H / 2)
    ax.set_aspect("equal")
    ax.set_title("I-Section Shape")


def create_output_plot(fitness_list, genome: Genome):
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    # Plot the I-section on the first subplot
    generate_section_plot(genome, axes[0])

    # Plot the fitness curve on the second subplot
    plot_fitness(fitness_list, axes[1])

    axes[0].minorticks_on()
    axes[1].minorticks_on()

    # Add minor grid lines
    axes[0].grid(True, which="minor", linestyle=":", color="gray")  # Minor grid on ax1
    axes[1].grid(True, which="minor", linestyle=":", color="gray")  # Minor grid on ax2

    # Optionally, add major grid lines for comparison
    axes[0].grid(True, which="major", linestyle="-", color="black")  # Major grid on ax1
    axes[1].grid(True, which="major", linestyle="-", color="black")  # Major grid on ax2

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


# run the evolution function
# NOTE: use partial to predefine missing arguments in signature
if __name__ == "__main__":
    # fitness_list = []
    (last_population, generation, fitness, fitness_list) = run_evolution(
        populate_func=partial(generate_population, population_size=10),
        fitness_func=fitness_function,
        selection_func=selection_pair,
        crossover_func=single_point_crossover,
        mutation_func=mutation,
        mutation_rate=0.2,
        max_generations=500,
        fitness_limit=1e50,
    )

    # Final answer plot
    create_output_plot(fitness_list, last_population[0])
