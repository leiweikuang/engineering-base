# This script solves the Knapsack problem using brute force and genetic algorithm
from email.mime import base
import random
from turtle import color
from typing import Callable
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import streamlit as st
import io, sys

# Wrap it in a streamlit app
# Title
st.subheader("Evolutionary Algorithm to Optimse Section design")

# Layout containers for top (graph) and bottom (terminal)
top_section = st.container()
bottom_section = st.container()

# Input Section
st.sidebar.header("Control Parameters")

# Base genome setting

genome_size_input = st.sidebar.slider("Population Size", 4, 20, 10, step=2)
mutation_rate_input = st.sidebar.slider(
    "Randomness of Initial Population", 0.0, 1.0, 0.1
)
child_mutation_spread_input = st.sidebar.slider("Child Mutation Rate", 0.0, 0.6, 0.3)
total_generations_input = st.sidebar.slider("Total Generations", 100, 5000, 2000)
min_width_input = st.sidebar.slider("Minimum Width of Flange (mm)", 200, 600, 400)
min_thickness_input = st.sidebar.slider("Minimum Web Thickness (mm)", 10, 20, 25)
area_limit_input = st.sidebar.slider("Area Limit (mm^2)", 10000, 30000, 20000)
re_draw_button = st.sidebar.button("Redraw samples")


# Redirect print output to streamlit
output = io.StringIO()
sys.stdout = output  # Redirect `print` to `output`


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


def check_genome_validity(
    genome: Genome,
) -> bool:
    # Check if the genome is valid
    H = genome[0]
    h = genome[1]
    a = genome[2]
    b = genome[3]

    if len(genome) != 4:
        return False
    if H < 0:
        return False
    if h < 0:
        return False
    if a < 0:
        return False
    if b < 0:
        return False
    if H - h < 0:
        return False
    if b < a:
        return False
    if calc_area(genome) <= 0:
        return False

    return True


def generate_genome(base_genome: Genome, spread: float = 0.9) -> Genome:
    # Generates a random genome of length 4,
    # use base genome to generate genome

    def generate_values():
        H = np.random.normal(1, spread) * base_genome[0]
        h = np.random.normal(1, spread) * base_genome[1]
        a = np.random.normal(1, spread) * base_genome[2]
        b = np.random.normal(1, spread) * base_genome[3]

        # Make sure numbers make sense
        if check_genome_validity(genome=[H, h, a, b]) is False:
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


def generate_population(
    base_genome: Genome,
    population_size: int,
) -> Population:
    # This function generates the population using the generate_genome function
    # until the population_size is reached
    population = []
    for _ in range(population_size):

        population.append(generate_genome(base_genome, mutation_rate_input))

    return population


def fitness_function(
    genome: Genome,
    area_limit: float = area_limit_input,
    min_width=min_width_input,
    min_thickness=min_thickness_input,
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
    parent1: Genome, parent2: Genome, spread: float = child_mutation_spread_input
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

    # create two new genomes based on parents.
    # use the average of the parents as the base
    H_base = (parent1[0] + parent2[0]) / 2
    h_base = (parent1[1] + parent2[1]) / 2
    a_base = (parent1[2] + parent2[2]) / 2
    b_base = (parent1[3] + parent2[3]) / 2

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
) -> tuple[Population, Population, int, int, list]:
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
            return (
                initial_population,
                population,
                i,
                fitness_func(population[0]),
                fitness_list,
            )

        next_population = population[0:2]
        remaning_population = range(int((len(population)) / 2) - 1)

        # Select parent pairs based on fitness. Note that we've already reserved the top 2 genomes
        for j in remaning_population:
            # Note that we can have overlapping parents, we just need to make sure the
            # next generation is the same length

            # If population value is not all zero, select based on weighting
            if sum(population_value_list) == 0:
                # Randomly select 2 genomes
                parents = random.sample(population, 2)
            else:
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

        if i == 1:
            initial_population = population.copy()

        # Replace the current population with the next population
        population = next_population

        fitness_list.append(fitness_func(population[0]))

    # After all is done, return the population and the generation number
    # sort it one last time in case we never reached the fitness limit
    population = sorted(population, key=fitness_func, reverse=True)

    with bottom_section:
        # Print current generation result
        value = round(calc_population_value(population, fitness_func)[0], 2)
        print(
            f"""
            Second moment of Inertia, Ix (mm^4): 
            {value:,.2f},
            Depth (mm):  
            {round(population[0][0], 2)},
            Area of Section (mm^2):  
            {round(calc_area(population[0]), 2)},
            Flange thickness (mm):  
            {round((population[0][0] - population[0][1]) / 2, 2)},
            Flange_width (mm):  
            {round(population[0][3], 2)},
            Web thickness (mm):  
            {round(population[0][2], 2)},
            """
        )

        # Reset stdout so further prints go to the console
        sys.stdout = sys.__stdout__

        # Display captured print output
        st.subheader("Final Generation")
        st.text(output.getvalue())

    return initial_population, population, i, fitness_func(population[0]), fitness_list


def plot_fitness(fitness_list, ax):
    """
    Plots the fitness values over generations.

    Parameters:
    fitness_list - A list of fitness values, where each index represents a generation.
    """
    generations = list(range(1, len(fitness_list) + 1))

    ax.plot(
        generations,
        fitness_list,
        linestyle="-",
        color="tab:blue",
        label="Fitness",
    )

    # Labels and title
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (Second Moment of Inertia)")
    ax.set_title(" Evolution of Second Moment of Inertia")


def generate_section_plot(genome: Genome, ax, title):
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
    ax.plot(x_coords, y_coords, "k", linewidth=2, color="tab:gray")
    ax.set_xlim(-b, b)
    ax.set_ylim(-H / 2, H / 2)
    ax.set_aspect("equal")
    ax.set_title(f"{title}")


def create_output_plot(fitness_list, genome1: Genome, genome2: Genome):
    # Put it in the top section
    with top_section:
        # Create subplots
        # Create a figure and gridspec
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2, figure=fig)

        # Create the axes for the subplots
        ax1 = fig.add_subplot(gs[0, 0])  # Upper left
        ax2 = fig.add_subplot(gs[1, 0])  # Lower left
        ax3 = fig.add_subplot(
            gs[:, 1]
        )  # Upper right (this takes half of the upper side)

        # Plot the I-section on the first subplot
        generate_section_plot(genome1, ax1, "First generation I Section")
        generate_section_plot(genome2, ax2, "Final generation I Section")

        # Plot the fitness curve on the second subplot
        plot_fitness(fitness_list, ax3)

        for ax in [ax1, ax2, ax3]:
            ax.minorticks_on()
            ax.grid(True, which="minor", linestyle=":", color="#CCCCCC")  # Minor grid
            ax.grid(True, which="major", linestyle="-", color="gray")  # Major grid

        # Adjust layout and show the plot
        # plt.tight_layout()

        graph_window_size = 800

        ax1.set_ylim(
            -graph_window_size, graph_window_size
        )  # y-limits for the upper left plot
        ax2.set_ylim(
            -graph_window_size, graph_window_size
        )  # y-limits for the lower left plot
        ax1.set_xlim(
            -graph_window_size, graph_window_size
        )  # x-limits for the upper left plot
        ax2.set_xlim(
            -graph_window_size, graph_window_size
        )  # x-limits for the lower left plot
        ax3.set_xlim(0)  # x-limits for the upper left plot
        ax3.set_ylim(0)  # x-limits for the lower left plot
        # plt.show()
        st.pyplot(fig)


# run the evolution function
# NOTE: use partial to predefine missing arguments in signature
if __name__ == "__main__":
    base_genome = [100, 50, 100, 500]
    fitness_list = []
    try:
        (initial_population, last_population, generation, fitness, fitness_list) = (
            run_evolution(
                populate_func=partial(
                    generate_population,
                    base_genome=[200, 20, 100, 200],
                    population_size=genome_size_input,
                ),
                fitness_func=fitness_function,
                selection_func=selection_pair,
                crossover_func=single_point_crossover,
                mutation_func=mutation,
                mutation_rate=mutation_rate_input,
                max_generations=total_generations_input,
                fitness_limit=1e50,
            )
        )
    except ValueError:
        (initial_population, last_population, generation, fitness, fitness_list) = (
            run_evolution(
                populate_func=partial(
                    generate_population,
                    base_genome=[200, 20, 100, 200],
                    population_size=genome_size_input,
                ),
                fitness_func=fitness_function,
                selection_func=selection_pair,
                crossover_func=single_point_crossover,
                mutation_func=mutation,
                mutation_rate=mutation_rate_input,
                max_generations=total_generations_input,
                fitness_limit=1e50,
            )
        )

    if re_draw_button:
        top_section.empty()
        bottom_section.empty()
        result = (
            initial_population,
            last_population,
            generation,
            fitness,
            fitness_list,
        ) = run_evolution(
            populate_func=partial(
                generate_population,
                base_genome=[200, 20, 100, 200],
                population_size=genome_size_input,
            ),
            fitness_func=fitness_function,
            selection_func=selection_pair,
            crossover_func=single_point_crossover,
            mutation_func=mutation,
            mutation_rate=mutation_rate_input,
            max_generations=total_generations_input,
            fitness_limit=1e50,
        )

    # Final answer plot
    create_output_plot(fitness_list, initial_population[0], last_population[0])
