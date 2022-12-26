import copy
import math
import random
import re
import string
from pprint import pprint

import awkward as ak
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

# %%


# blueprints = np.loadtxt('test_input', dtype=object, delimiter='\n')
blueprints = np.loadtxt('input', dtype=object, delimiter='\n')
pprint(blueprints[0])
blueprints = np.array([re.findall('\d+', blueprint) for blueprint in blueprints]).astype(int)

# %%

# simulated annealing


@njit
def simulated_annealing(energy_function, n_min, initial_state, costs, Tmin, temperature, max_iterations):
    T_scaling = (Tmin / temperature) ** (1 / max_iterations)
    current_state = initial_state
    best_state = initial_state.copy()
    best_energy = 0
    current_energy = energy_function(current_state, costs, n_min)
    for i in range(max_iterations):
        # Generate a random new state by changing one element of the current state
        new_state = current_state.copy()
        index = np.random.randint(0, len(new_state))
        new_state[index] = np.random.randint(0, 4)

        # Calculate the energy of the new state
        new_energy = energy_function(new_state, costs, n_min)

        # Decide whether to accept the new state based on the difference in energy and the current temperature
        energy_difference = new_energy - current_energy
        if energy_difference > 0:
            # Accept the new state with a probability equal to e^(-energy_difference / temperature)
            probability = math.exp(-energy_difference / temperature)
            if np.random.uniform(0, 1) < probability:
                current_state = new_state
                current_energy = new_energy
        else:
            # Always accept a state that has lower energy
            current_state = new_state
            current_energy = new_energy

        # Decrease the temperature according to a cooling schedule
        temperature *= T_scaling
        if current_energy < best_energy:
            best_energy = current_energy
            best_state = current_state
        if i % 10_000_000 == 0:
            print(i, best_energy)

    return best_state, best_energy


@njit
def energy_function(state, costs, n_min):
    robots = np.array([1, 0, 0, 0])
    resources = np.array([0, 0, 0, 0])
    minutes = 0
    for i, action in enumerate(state):
        available = False
        while not available:
            # check for available resources at start of minute
            available = (costs[:, action] <= resources[:-1]).sum() == 3
            resources += robots  # increase resources
            minutes += 1
            if minutes >= n_min:
                return -resources[3] - i / len(state)
        # build robot
        robots[action] += 1
        resources[:-1] -= costs[:, action]

    return -resources[3] - i / len(state)


# %%

part2 = True

if part2:
    n_min = 32
    max_iterations = 100_000_000
    sel_blueprints = blueprints[:3]
else:
    n_min = 24
    max_iterations = 10_000_000
    sel_blueprints = blueprints

best_geos = []
for b, blueprint in enumerate(sel_blueprints):
    print('blueprint', b)
    costs = np.array([[blueprint[1], blueprint[2], blueprint[3], blueprint[5]],
                      [0, 0, blueprint[4], 0],
                      [0, 0, 0, blueprint[6]]])

    initial_state = np.random.randint(0, 4, n_min)
    best_order, best_geo = simulated_annealing(energy_function, n_min, initial_state, costs,
                                               Tmin=0.001,
                                               temperature=50,
                                               max_iterations=max_iterations)
    print(best_order, best_geo)
    best_geos.append(best_geo)

# %%

if not part2:
    print('part 1:')
    print(np.sum(-np.ceil(best_geos).astype(int) * np.arange(1, len(blueprints) + 1)))
    # 978
else:
    print('part 2:')
    print(int(np.product(-np.ceil(best_geos[:3]))))
    # 15939

# %%
# 56, 62
list(np.ceil(best_geos).astype(int))
# best = [0, -15, -1, -3, -3, 0, -2, -5, -9, -2, -5, 0, 0, -1, 0, 0, -13, -5, 0, -2, -1, 0, -1, 0, -4, 0, 0, -5, -1, -1]

# %%
# %%
# %%

# genetic algorithm


def generate_random_solution(items, possible_solutions):
    # Generate a random order for the items
    order = np.random.choice([0, 1, 2, 3], len(items))
    # if list(order[:len(possible_solutions[0])]) not in possible_solutions:
    #     order[:len(possible_solutions[0])] = random.choice(possible_solutions)
    return order


def genetic_algorithm(order, costs, loss, possible_loss, population_size=50, num_generations=100,
                      mutation_rate=0.05, tournament_size=3, keep_best=3, possible_solutions=[[0], [1]]):
    # Initialize the population with random orders
    population = []
    for i in range(population_size):
        population.append(generate_random_solution(order, possible_solutions))

    for generation in tqdm(range(num_generations)):
        # Evaluate the loss for each order in the population
        losses = [possible_loss(order, costs, possible_solutions) for order in population]

        # Select the fittest orders using tournament selection
        fittest_orders = []
        for _ in range(population_size):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_losses = [losses[tournament_i] for tournament_i in tournament_indices]
            fittest_orders.append(population[tournament_indices[tournament_losses.index(min(tournament_losses))]])

        # Create the next generation of orders using crossover and mutation
        population = [population[losses.index(loss)] for loss in sorted(losses)[:keep_best]]
        for _ in range(population_size - keep_best):
            parent1, parent2 = random.sample(fittest_orders, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate * (1 - generation / num_generations * 19 / 20))
            population.append(child)

            # if list(child[:len(possible_solutions[0])]) not in possible_solutions:
            #     child[:len(possible_solutions[0])] = random.choice(possible_solutions)

        if generation % 500 == 0:
            print(generation, min(losses))

    # Return the fittest order from the final generation
    final_losses = [possible_loss(order, costs, possible_solutions) for order in population]
    return population[np.argmin(final_losses)], min(final_losses)


def crossover(order1, order2):
    # Perform crossover using a two-point crossover method
    crossover_point = random.randint(1, int(len(order1) * 0.6 - 1))
    child = np.concatenate([order1[:crossover_point], order2[crossover_point:]])

    # crossover_points = sorted(np.random.randint(1, len(order1) - 1, 2))
    # child = np.concatenate([order1[:crossover_points[0]],
    #                         order2[crossover_points[0]:crossover_points[1]],
    #                         order1[crossover_points[1]:]])

    # to_replace = np.random.random(len(order1)) > 0.5
    # child = order1.copy()
    # child[to_replace] = order2[to_replace]
    return child


def mutate(order, mutation_rate):
    to_replace = np.random.random(len(order)) < mutation_rate
    order[to_replace] = np.random.choice([0, 1, 2, 3], to_replace.sum())

    # if list(order[:len(possible_solutions[0])]) not in possible_solutions:
    #     order[:len(possible_solutions[0])] = random.choice(possible_solutions)

    # for index1 in indices:
    #     if random.random() < mutation_rate:
    #         order[index1] = random.choice([0, 1, 2, 3])
    return order

# %%


@njit
def possible_loss(gene, costs, possible_solutions):
    return np.array([loss(np.append(possible_solution, gene), costs) for possible_solution in possible_solutions]).min()


@njit
def loss(gene, costs):
    robots = np.array([1, 0, 0, 0])
    resources = np.array([0, 0, 0, 0])
    minutes = 0
    for i, action in enumerate(gene):
        available = False
        while not available:
            # check for available resources at start of minute
            available = (costs[:, action] <= resources[:-1]).sum() == 3

            # increase resources
            for key, value in enumerate(robots):
                resources[key] += value

            minutes += 1
            if minutes >= 24:
                return -resources[3] - i / len(gene)

        # build robot
        robots[action] += 1
        resources[:-1] -= costs[:, action]

    return -resources[3] - i / len(gene)
    # - 0.25 * (np.diff(np.array([split.sum() for split in np.split(best_order, 4)])) > 0).sum() / 3


def find_solutions(costs, robots=np.array([1, 0, 0, 0]), resources=np.array([0, 0, 0, 0]), gene=[], genes=[], minutes=0, action=None, return_ore=False):
    if action is not None:
        available = False
        while not available:
            # check for available resources at start of minute
            available = (costs[slice(None), action] <= resources[:-1]).sum() == 3

            # increase resources
            resources += robots
            # for key, value in enumerate(robots):
            #     resources[key] += value

            minutes += 1
            if minutes >= 24:
                if (resources[3] > 0):
                    if return_ore:
                        # gives ores for all possible order lengths when hitting max minutes
                        genes.append([resources[3], minutes, gene])
                    # else:
                    #     genes.append(gene)
                return genes

        robots[action] += 1
        resources[:-1] -= costs[slice(None), action]
        gene.append(action)

    if len(gene) >= 5:
        resources += robots * (24 - minutes)
        if return_ore:
            if resources[3] > 0:
                genes.append([resources[3], minutes, gene])
        else:
            # enough resources in the end to buy n_rob robots
            # n_rob = 1.0
            # if (resources[3] > 0) or ((np.divide(resources[:-1], costs[slice(None), 3], where=costs[slice(None), 3] > 0, out=np.ones(3)) >= n_rob).sum() == 3):
            genes.append(gene)
            # genes.append([resources[3], minutes, gene])
    else:
        for action in range(4):
            genes = find_solutions(costs, robots.copy(), resources.copy(), gene.copy(), genes.copy(),
                                   minutes=minutes, action=action, return_ore=return_ore)

    return genes

# %%


# brute force
solutions = find_solutions(costs, return_ore=True)
len(solutions)
solutions
solutions = find_solutions(costs, return_ore=False)
len(solutions)
solutions
sorted(solutions, reverse=True, key=lambda x: x[0])[:5]

# %%

use_genetic = True
best_geos = []
for b, blueprint in enumerate(blueprints):
    # if b != 22:
    #     continue

    print('blueprint', b)
    costs = np.array([[blueprint[1], blueprint[2], blueprint[3], blueprint[5]],
                      [0, 0, blueprint[4], 0],
                      [0, 0, 0, blueprint[6]]])

    if use_genetic:
        order = np.array([0] * (24 - 5))  # not relevant for population
        # possible_solutions = find_solutions(costs)
        possible_solutions = np.array(find_solutions(costs))
        print(len(possible_solutions))
        if len(possible_solutions) == 0:
            best_geo, best_order = 0, None
        else:
            best_order, best_geo = genetic_algorithm(order, costs, loss, possible_loss, population_size=150, num_generations=1_000,
                                                     mutation_rate=1.0, tournament_size=15, keep_best=1, possible_solutions=possible_solutions)
    else:
        solutions = find_solutions(costs, return_ore=True)
        if len(solutions) > 0:
            best_geo, n_minutes, best_order = sorted(solutions, reverse=True, key=lambda x: x[0])[0]
        else:
            best_geo, best_order = 0, None
        print(len(solutions), n_minutes)

    print(best_order, best_geo)
    best_geos.append(best_geo)
# [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 2, 3, 3, 3, 1, 2, 2, 2, 2, 2, 0]
# %%

print('part 1:')
print(np.sum(-np.ceil(best_geos).astype(int) * np.arange(1, len(blueprints) + 1)))

list(np.ceil(best_geos).astype(int))
# 797
# 806
# 817
# 850
# 885
# 941
# 964
# 964

# %%

best = [0, -15, -1, -3, -3, 0, -2, -5, -9, -2, -5, 0, 0, 0, 0, 0, -13, -5, 0, -2, -1, 0, -1, 0, -4, 0, 0, -5, -1, -1]
print('part1:', np.sum(-np.array(best) * np.arange(1, len(blueprints) + 1)))

# %%
# %%
# %%
