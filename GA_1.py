import random

# Parameters
POPULATION_SIZE = 50
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.05
NUM_GENERATIONS = 100
GENE_LENGTH = 10  # Binary string length

# Fitness Function
def fitness(individual):
    return sum(individual)  # Example: Maximize sum of bits

# Initialize Population
def initialize_population():
    return [[random.randint(0, 1) for _ in range(GENE_LENGTH)] for _ in range(POPULATION_SIZE)]

# Selection
def tournament_selection(population, scores, k=3):
    selected = random.sample(list(zip(population, scores)), k)
    return max(selected, key=lambda x: x[1])[0]

# Crossover
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, GENE_LENGTH - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1, parent2

# Mutation
def mutate(individual):
    return [1 - gene if random.random() < MUTATION_RATE else gene for gene in individual]

# Main Genetic Algorithm Loop
population = initialize_population()

for generation in range(NUM_GENERATIONS):
    scores = [fitness(ind) for ind in population]
    next_generation = []
    
    for _ in range(POPULATION_SIZE // 2):
        parent1 = tournament_selection(population, scores)
        parent2 = tournament_selection(population, scores)
        offspring1, offspring2 = crossover(parent1, parent2)
        next_generation.extend([mutate(offspring1), mutate(offspring2)])
    
    population = next_generation
    best_individual = max(population, key=fitness)
    print(f"Generation {generation + 1}: Best Fitness = {fitness(best_individual)}")

# Result
print("Best Individual:", best_individual)
