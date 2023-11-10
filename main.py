import random
import numpy as np
import math
import matplotlib.pyplot as plt

# Define your parameters
num_groups = 4
subjects_per_group = 5
days_per_week = 5
slots_per_day = 1 # math.ceil(subjects_per_group / days_per_week)

# Create a list of subjects for each group
groups = {f"Group_{i}": [f"Subject_{i}_{j}" for j in range(subjects_per_group)] for i in range(num_groups)}

# Create a list of lecturers
lecturers = [f"Lecturer_{i}" for i in range(subjects_per_group)]

# Function to plot fitness scores
def plot_fitness_scores(max_scores, avg_scores):
    generations = range(1, len(max_scores) + 1)
    plt.plot(generations, max_scores, label='Max Fitness')
    plt.plot(generations, avg_scores, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.legend()
    plt.title('Genetic Algorithm Fitness Progress')
    plt.show()


# Generate initial population
def generate_schedule():
    schedule = {group: {day: [None] * slots_per_day for day in range(days_per_week)} for group in groups}

    for group, subjects in groups.items():
        # Choose subjects_per_group positions for the group
        selected_positions = random.sample(range(slots_per_day * days_per_week), subjects_per_group)

        for position in selected_positions:
            day = position // slots_per_day
            slot = position % slots_per_day
            lecturer = random.choice(lecturers)
            schedule[group][day][slot] = lecturer

    return schedule


# Fitness function
def fitness(schedule):
    penalty = 1

    for group, subjects in groups.items():
        lectures_count = sum(1 for day in range(days_per_week) for slot in range(slots_per_day) if schedule[group][day][slot] is not None)
        if lectures_count != subjects_per_group:
            penalty += 5

        # Check for duplicate lectures in the same group
        seen_lectures = set()
        for day in range(days_per_week):
            for slot in range(slots_per_day):
                lecture = schedule[group][day][slot]
                if lecture is not None:
                    if lecture in seen_lectures:
                        penalty += 0.2
                    seen_lectures.add(lecture)

    # Check if the lecturer teaches two groups at the same time
    for lecturer in lecturers:
        lecturer_schedule = {group: [] for group in groups}
        for group, daily_schedule in schedule.items():
            for day, classes in daily_schedule.items():
                for time_slot, assigned_lecturer in enumerate(classes):
                    if assigned_lecturer == lecturer:
                        lecturer_schedule[group].append((day, time_slot))

        for group1, schedule1 in lecturer_schedule.items():
            for group2, schedule2 in lecturer_schedule.items():
                if group1 != group2:
                    for day1, time_slot1 in schedule1:
                        for day2, time_slot2 in schedule2:
                            if day1 == day2 and time_slot1 == time_slot2:
                                penalty += 0.1

    # Return a higher value for better schedules (lower penalty)
    return 1 / penalty




# Crossover function
def crossover(parent1, parent2):
    crossover_point = random.choice(list(parent1.keys()))
    child = parent1.copy()

    for group in groups:
        if group < crossover_point:
            child[group] = parent1[group].copy()
        else:
            child[group] = parent2[group].copy()

    return child


# Mutation function
def mutate(schedule):
    mutated_schedule = schedule.copy()
    i = 0
    while True:
        group_to_mutate = random.choice(list(mutated_schedule.keys()))
        day_to_mutate = random.choice(list(mutated_schedule[group_to_mutate].keys()))
        curr_lec = [item for sublist in list(mutated_schedule[group_to_mutate].values()) for item in sublist]
        extra_lecturers = [lecturer for lecturer in lecturers if lecturer not in curr_lec]
        i += 1
        if len(extra_lecturers) > 0 or i > num_groups * days_per_week:
            break
    if len(extra_lecturers) == 0:
        return mutated_schedule
    elif len(extra_lecturers) < slots_per_day:
        lecturer_to_add = [random.choice(extra_lecturers) for _ in range(len(extra_lecturers))]
        for i in range(slots_per_day - len(lecturer_to_add)):
            lecturer_to_add.append(None)
        random.shuffle(lecturer_to_add)
    else:
        lecturer_to_add = random.sample(extra_lecturers, slots_per_day)
    mutated_schedule[group_to_mutate][day_to_mutate] = lecturer_to_add

    return mutated_schedule


def selection(popula, idx, n):
    offspring = []
    for i in range(n):
        i1 = i2 = i3 = i4 = 0
        while i1 in [i2, i3, i4] or i2 in [i1, i3, i4] or i3 in [i1, i2, i4] or i4 in [i1, i2, i3]:
            i1, i2, i3, i4 = random.randint(0, n - 1), random.randint(0, n - 1), random.randint(0,
                                                                                                n - 1), random.randint(
                0, n - 1)
        max_idx = np.argmax([idx[i1], idx[i2], idx[i3], idx[i4]])
        offspring.append(popula[[i1, i2, i3, i4][max_idx]])

    return offspring


# Genetic Algorithm
def genetic_algorithm(population_size, generations):
    population = [generate_schedule() for _ in range(population_size)]

    max_fitness_scores = []
    avg_fitness_scores = []
    for generation in range(generations):
        # Evaluate fitness of each schedule in the population
        fitness_scores = [fitness(schedule) for schedule in population]
        max_fitness_scores.append(max(fitness_scores))
        avg_fitness_scores.append(np.mean(fitness_scores))
        if max(fitness_scores) == 1:
            break

        # Select the top performers
        top_indices = np.argsort(fitness_scores)[-population_size // 2:]
        #
        # Create the next generation through crossover and mutation
        next_generation = [population[i] if random.random() > mutation_rate else mutate(population[i]) for i in top_indices]
        # next_generation = selection(population, fitness_scores, population_size)
        for _ in range(population_size // 2):
            parent1 = population[random.choice(top_indices)]
            parent2 = population[random.choice(top_indices)]
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
            next_generation.append(child)

        # Replace the old generation with the new one
        population = next_generation

    plot_fitness_scores(max_fitness_scores, avg_fitness_scores)
    # Return the best schedule found
    best_schedule_index = np.argmax(fitness_scores)
    return population[best_schedule_index]


# Parameters
population_size = 100
generations = 200
mutation_rate = 0.2

# Run the genetic algorithm
best_schedule = genetic_algorithm(population_size, generations)

# Print the best schedule
for group, schedule in best_schedule.items():
    print(f"Schedule for {group}:")
    for day, classes in schedule.items():
        print(f"Day {day + 1}: {classes}")
