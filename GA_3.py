import random
import math

# Problem Parameters
NUM_VESSELS = 5
NUM_BERTHS = 3
ARRIVAL_TIMES = [1, 2, 3, 4, 5]  # Arrival times of vessels
CAPACITIES = [100, 200, 150, 100, 300]  # Cargo capacities of vessels
DISCHARGE_RATES = [50, 60, 40]  # Per day discharge rates for each berth

# Ongoing vessel discharges at berths
CURRENT_BERTHS = [[1, 2], [2, 3], None]  # [vessel_id, remaining_time] for each berth
POPULATION_SIZE = 10
NUM_GENERATIONS = 50
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

# Calculate Processing Time for Each Vessel at Each Berth
PROCESSING_TIMES = [
    [math.ceil(CAPACITIES[i] / DISCHARGE_RATES[j]) for j in range(NUM_BERTHS)]
    for i in range(NUM_VESSELS)
]

# Initialize Berth Availability
berth_end_times = [0] * NUM_BERTHS
for i, berth in enumerate(CURRENT_BERTHS):
    if berth:
        vessel_id, remaining_time = berth
        berth_end_times[i] = remaining_time  # Update berth availability

# Initialize Population
def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        chromosome = []
        for i in range(NUM_VESSELS):
            berth = random.randint(0, NUM_BERTHS - 1)
            start_time = max(ARRIVAL_TIMES[i], berth_end_times[berth])
            chromosome.append([berth, start_time])
        population.append(chromosome)
    return population

# Fitness Function
def fitness(chromosome):
    total_waiting_time = 0
    berth_end_times_copy = berth_end_times[:]  # Copy to avoid modifying global state

    for i, (berth, start_time) in enumerate(chromosome):
        processing_time = PROCESSING_TIMES[i][berth]
        start_time = max(berth_end_times_copy[berth], start_time)
        waiting_time = max(0, start_time - ARRIVAL_TIMES[i])
        total_waiting_time += waiting_time
        berth_end_times_copy[berth] = start_time + processing_time  # Update berth availability

    return 1 / (1 + total_waiting_time)

# Selection
def tournament_selection(population, scores, k=3):
    selected = random.sample(list(zip(population, scores)), k)
    return max(selected, key=lambda x: x[1])[0]

# Crossover
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, NUM_VESSELS - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1, parent2

# Mutation
def mutate(chromosome):
    for gene in chromosome:
        if random.random() < MUTATION_RATE:
            berth = random.randint(0, NUM_BERTHS - 1)
            start_time = max(ARRIVAL_TIMES[chromosome.index(gene)], berth_end_times[berth])
            gene[0], gene[1] = berth, start_time
    return chromosome

# Main GA Loop
population = initialize_population()

for generation in range(NUM_GENERATIONS):
    scores = [fitness(ind) for ind in population]
    next_population = []

    for _ in range(POPULATION_SIZE // 2):
        parent1 = tournament_selection(population, scores)
        parent2 = tournament_selection(population, scores)
        offspring1, offspring2 = crossover(parent1, parent2)
        next_population.extend([mutate(offspring1), mutate(offspring2)])

    population = next_population
    best_individual = max(population, key=fitness)
    print(f"Generation {generation + 1}: Best Fitness = {fitness(best_individual)}")

# Best Solution
print("Best Allocation:", best_individual)