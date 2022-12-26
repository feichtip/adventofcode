import collections
import copy
import itertools
import math
import random
import re
import string
from ast import literal_eval
from collections import ChainMap
from pprint import pprint
from random import randrange

import awkward as ak
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%

test = False
part2 = True

if test:
    valves = np.loadtxt('test_input', dtype=object, delimiter='\n')
else:
    valves = np.loadtxt('input', dtype=object, delimiter='\n')


rates = {re.findall('[A-Z][A-Z]', valve)[0]: int(re.search(r'rate=(\d+)', valve).group(1)) for valve in valves}
valves = {re.findall('[A-Z][A-Z]', valve)[0]: re.findall('[A-Z][A-Z]', valve)[1:] for valve in valves}

# %%

# n_valves = len(list(valves))
# num = 2**n_valves - 1
# binary = format(num, 'b')

# always open the graph
if part2:
    time = 26
else:
    time = 30
graph = {}
for minute in range(time + 1):
    # update_dict = {f'{valve}{minute}': {f'{target}{minute + 2}': - rates[target] * (30 - minute) for target in targets} for valve, targets in valves.items()}
    update_dict = {f'{valve}{minute}': {**{f'{target}{minute + 1}': 0 for target in targets},
                                        **({f'{valve}{minute + 1}': - rates[valve] * (time - minute - 1)} if (rates[valve] > 0) else {})} for valve, targets in valves.items()}
    if minute == time:
        graph.update({f'{valve}{minute}': {} for valve, targets in valves.items()})
    else:
        graph.update(update_dict)

predecessor_list = [None]
distance_list = [None]

# %%


def draw_graph(graph, predecessor=None, distance=None, ax=None):
    edges = [[node, neighbour] for node, value in graph.items() for neighbour in value.keys()]
    G = nx.Graph()
    G.add_edges_from(edges)
    edges = list(G.edges)
    nodes = list(G.nodes)
    if predecessor is not None and distance is not None:
        path_edges = [(key, value) for key, value in predecessor.items() if value is not None]
        path_edges += [(value, key) for key, value in predecessor.items() if value is not None]

        last_node, best_len = list(distance.items())[np.argmin(list(distance.values()))]
        print(f'last node: {last_node} with len: {-best_len}')
        ancestor_list = ancestors(predecessor, last_node)
        path_edges_1 = [path_edge for path_edge in path_edges if (path_edge[0] in ancestor_list) and (path_edge[1] in ancestor_list)]

        # 2nd best path
        distance_2 = {key: value for key, value in distance.items() if (key not in ancestor_list) and (int(key[2:]) <= int(last_node[2:]))}
        last_node_2, best_len_2 = list(distance_2.items())[np.argmin(list(distance_2.values()))]
        print(f'last node: {last_node_2} with len: {-best_len_2}')
        ancestor_list_2 = ancestors(predecessor, last_node_2)
        path_edges_2 = [path_edge for path_edge in path_edges if (path_edge[0] in ancestor_list_2) and (path_edge[1] in ancestor_list_2)]
    else:
        path_edges_1 = []
        path_edges_2 = []
        last_node = ''
    edge_color = ['firebrick' if edge in path_edges_1 else ('peru' if edge in path_edges_2 else 'k') for edge in edges]
    width = [3 if edge in path_edges_1 else (2 if edge in path_edges_2 else 1) for edge in edges]
    node_color = ['firebrick' if node == last_node else 'powderblue' for node in nodes]
    pos = {f'{valve}{min}': (min / 10, i / 10) for min in range(time + 1) for i, valve in enumerate(list(valves))}
    nx.draw_networkx(G, pos=pos, edge_color=edge_color, node_color=node_color, node_size=700,
                     alpha=0.9, linewidths=0, style='solid', width=width, ax=ax)
    # plt.show()


# %%

if test:
    predecessor = predecessor_list[-1]
    distance = distance_list[-1]
    plt.figure(figsize=(25, 10))
    draw_graph(graph, predecessor, distance)
    plt.show()

# %%

keys = list(graph.keys())
keys = sorted(keys, key=lambda x: [int(x[2:]), x[:2]], reverse=False)
graph = {key: graph[key] for key in keys}

# %%

to_open = [key for key, rate in rates.items() if rate > 0]
math.factorial(len(to_open))

# %%

# simplify graph, remove 0 weight edges
for i in range(50):
    old_graph = graph.copy()
    # remove all 0s except when turning on valve
    # graph = {node: dict(ChainMap(*[graph[key] if (rates[key[:2]] == 0) else {key: item} for key, item in neighbours.items()]))
    #          for node, neighbours in graph.items()}

    # remove all 0s
    graph = {node: dict(ChainMap(*[graph[key] if (item == 0) else {key: item} for key, item in neighbours.items()]))
             for node, neighbours in graph.items()}

    graph = {node: {key: item for key, item in neighbours.items() if
                    sorted([neig for neig in neighbours.keys() if neig[:2] == key[:2]], key=lambda x: [int(x[2:]), x[:2]]).index(key) == 0}
             for node, neighbours in graph.items()}
    print(i, len(graph))
    if graph == old_graph:
        break

graph = {node: neighbours for node, neighbours in graph.items() if (rates[node[:2]] > 0) or node == 'AA0'}

# %%


def loss(order, return_valve_order=False):
    global graph, part2
    best_flow = 0
    best_order = []
    if part2:
        splits = [[order[:i], order[i:]] for i in range(1, len(order))]
    else:
        splits = [[order]]
    for split in splits:
        total_flow = 0
        valve_order = []
        for s, part in enumerate(split):
            neighbours = graph['AA0']
            for o in part:
                next_item = [[key, flow] for key, flow in neighbours.items() if key[:2] == o]
                if len(next_item) == 0:
                    continue
                else:
                    assert (len(next_item) == 1)
                    next_valve, gained_flow = next_item[0]
                neighbours = graph[next_valve]
                valve_order.append((s, next_valve))
                total_flow += gained_flow
                # print(next_valve, neighbours, gained_flow)
        # print(split, total_flow)
        if total_flow < best_flow:
            best_flow = total_flow
            best_order = valve_order
    if return_valve_order:
        return best_flow, best_order
    return best_flow


# brute force, only with test
if test:
    best_total_flow = 0
    for order in tqdm(list(itertools.permutations(to_open))):
        total_flow = loss(order)
        if total_flow < best_total_flow:
            best_total_flow = total_flow
            best_valve_order = order

    print(best_total_flow, best_valve_order)

    print(loss(best_valve_order, return_valve_order=True))
else:
    print(loss(to_open, return_valve_order=True))


# %%


def generate_random_solution(items):
    # Generate a random order for the items
    order = items.copy()
    random.shuffle(order)
    return order


def genetic_algorithm(order, loss, population_size=50, num_generations=100,
                      mutation_rate=0.1, tournament_size=3, keep_best=3):
    # Initialize the population with random orders
    population = []
    for i in range(population_size):
        population.append(generate_random_solution(order))

    for generation in tqdm(range(num_generations)):
        # Evaluate the loss for each order in the population
        losses = [loss(order) for order in population]

        # Select the fittest orders using tournament selection
        fittest_orders = []
        for _ in range(population_size):
            tournament_orders = random.sample(population, tournament_size)
            tournament_losses = [loss(order) for order in tournament_orders]
            fittest_orders.append(tournament_orders[tournament_losses.index(min(tournament_losses))])

        # Create the next generation of orders using crossover and mutation
        population = [population[losses.index(loss)] for loss in sorted(list(set(losses)))[:keep_best]]
        for _ in range(population_size - keep_best):
            parent1, parent2 = random.sample(fittest_orders, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            population.append(child)

        if generation % 50 == 0:
            print(generation, min(losses))

    # Return the fittest order from the final generation
    final_losses = [loss(order) for order in population]
    return population[np.argmin(final_losses)], min(final_losses)


def random_insert(lst, item):
    lst.insert(randrange(len(lst) + 1), item)


def crossover(order1, order2):
    # Perform crossover using a single-point crossover method
    crossover_point = random.randint(1, len(order1) - 1)
    child = order1[:crossover_point] + order2[crossover_point:]
    # remove duplicates
    child = list(dict.fromkeys(child))

    # Ensure that the child order contains all elements from the original list
    for element in order1 + order2:
        if element not in child:
            random_insert(child, element)

    return child


def mutate(order, mutation_rate):
    # Swap two items in the order
    index1 = random.randint(0, len(order) - 1)
    index2 = random.randint(0, len(order) - 1)
    order[index1], order[index2] = order[index2], order[index1]
    return order


# %%

# part1: (-1850, ['IP3', 'UF7', 'ZJ11', 'OD14', 'SB18', 'CW21', 'WK24'])
# part2: (-2306, [(0, 'IP3'), (0, 'UF7'), (0, 'ZJ11'), (0, 'OD14'), (0, 'SB18'), (0, 'CW21'), (0, 'WK24'), (1, 'DV3'), (1, 'PD7'), (1, 'CL11'), (1, 'DR14'), (1, 'CF17')])

optimal_order, optimal_flow = genetic_algorithm(to_open, loss, population_size=250, num_generations=200,
                                                mutation_rate=0.4, tournament_size=5, keep_best=10)
print(optimal_flow, optimal_order)
print(loss(optimal_order, return_valve_order=True))

# %%
# %%
# %%
# %%

# Bellman–Ford


def ancestors(predecessor, target):
    if target is None:
        return []
    else:
        return [target] + ancestors(predecessor, predecessor.get(target))


def repeats(predecessor, target):
    ancestor_list = [ancestor[:2] for ancestor in ancestors(predecessor, target)]   # only get letters
    # if repeated visits
    repeating = np.any([val > 1 for val in collections.Counter([ancestor_list[i]
                                                                for i in range(len(ancestor_list) - 1) if (ancestor_list[i] == ancestor_list[i + 1])]).values()])
    return repeating


def bellman_ford(graph, source, animate=True):
    distance, predecessor = dict(), dict()
    for node in graph:
        distance[node], predecessor[node] = float('inf'), None
    distance[source] = 0

    keys = list(graph.keys())

    distance_list = []
    predecessor_list = []
    graph_instance = graph
    for _ in tqdm(range(31)):
        for i in np.arange(0, 300, 1):
            if i <= 30:
                # graph_instance = {key: graph[key] for key in keys if int(key[2:]) == i}
                graph_instance = {key: graph[key] for key in keys if int(key[2:]) == random.randint(0, 30)}
            else:
                graph_instance = {key: graph[key] for key in keys if int(key[2:]) == random.randint(0, 30)}

            for node, neighbours in graph_instance.items():
                for neighbour, neighbour_weight in neighbours.items():
                    # If the distance between the node and the neighbour is lower than the current, store it
                    if distance[neighbour] > distance[node] + neighbour_weight:
                        distance_before, predecessor_before = distance[neighbour], predecessor[neighbour]

                        # store only if there are no repeats
                        distance[neighbour], predecessor[neighbour] = distance[node] + neighbour_weight, node
                        # revert if it repeats
                        if repeats(predecessor, neighbour):
                            # distance[neighbour], predecessor[neighbour] = float('inf'), None
                            distance[neighbour], predecessor[neighbour] = distance_before, predecessor_before
            distance_list.append(distance.copy())
            predecessor_list.append(predecessor.copy())
            # print(int(-np.min(list(distance.values()))))

    # Check for negative weight cycles
    # for node, neighbours in graph.items():
    #     for neighbour, neighbour_weight in neighbours.items():
    #         assert distance[neighbour] <= distance[node] + neighbour_weight, "Negative weight cycle."

    return distance_list, predecessor_list


distance_list, predecessor_list = bellman_ford(graph, source='AA0', animate=True)
last_node, best_len = list(distance_list[-1].items())[np.argmin(list(distance_list[-1].values()))]
print(f'last node: {last_node} with len: {-best_len}')

distance_list[-1]

print(len(distance_list))

fig, ax = plt.subplots(figsize=(25, 10))
draw_graph(graph, predecessor_list[-1], distance_list[-1])
plt.show()

# %%


G = nx.Graph()
edges = [[key, value] for key, value in predecessor_list[-1].items() if value is not None]
G.add_edges_from(edges)
plt.figure(figsize=(25, 10))

# labels = {f'{valve}{min}': distance.get(f'{valve}{min}') for min in range(time) for i, valve in enumerate(list(valves))}
all_edges = list(np.array(edges).flatten())
pos = {f'{valve}{min}': (min / 10, i / 10) for min in range(time + 1) for i, valve in enumerate(list(valves)) if f'{valve}{min}' in all_edges}
# nx.draw_networkx(G, labels={key: f'{key} {value}' for key, value in distance.items() if key in all_edges})
nx.draw_networkx(G, pos=pos)
plt.show()

# %%

fig, ax = plt.subplots(figsize=(25, 10))


def update(frame):
    predecessor = predecessor_list[frame]
    distance = distance_list[frame]
    ax.clear()
    draw_graph(graph, predecessor, distance, ax=ax)


# ani = animation.FuncAnimation(fig, update, frames=range(len(distance_list)), repeat=False)
ani = animation.FuncAnimation(fig, update, frames=np.arange(0, len(distance_list), 1))
# ani.save('animation.gif', writer='imagemagick')
ani.save("animation.mp4")
plt.show()

# %%

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Create a figure and an axis
fig, ax = plt.subplots()

# Set the axis limits
ax.set_xlim((0, 2))
ax.set_ylim((-2, 2))

# Initialize the line plot
line, = ax.plot([], [], lw=2)

# Define a function that updates the plot for each frame


def update(num):
    # Compute the x and y data for the next frame
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * num))
    # Set the x and y data for the line plot
    line.set_data(x, y)
    return line,


# Create an animation using the update function and a fixed interval
ani = animation.FuncAnimation(fig, update, frames=100, interval=20, blit=True)

ani.save('animation.gif', writer='imagemagick')
# Show the plot
plt.show()

# %%
