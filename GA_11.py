import random

# Problem Parameters
NUM_VESSELS = 5
NUM_BERTHS = 3
ARRIVAL_TIMES = [1, 2, 3, 4, 5]
PROCESSING_TIMES = [3, 2, 4, 1, 2]
POPULATION_SIZE = 10
NUM_GENERATIONS = 50
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

# Generate Initial Population
def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        chromosome = [[random.randint(0, NUM_BERTHS - 1), random.randint(ARRIVAL_TIMES[i], ARRIVAL_TIMES[i] + 10)]
                      for i in range(NUM_VESSELS)]
        population.append(chromosome)
    return population

# Fitness Function
def fitness(chromosome):
    total_waiting_time = 0
    for i, (berth, start_time) in enumerate(chromosome):
        total_waiting_time += max(0, start_time - ARRIVAL_TIMES[i])
    return 1 / (1 + total_waiting_time)  # Lower waiting time is better

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
            gene[0] = random.randint(0, NUM_BERTHS - 1)  # Change berth
            gene[1] = random.randint(ARRIVAL_TIMES[chromosome.index(gene)], ARRIVAL_TIMES[chromosome.index(gene)] + 10)
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
