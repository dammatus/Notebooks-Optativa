from matplotlib import pyplot as plt
import numpy as np
import random
import os

D = 10

def ackley_1(x):
    """
    Función de Ackley 1.

    Parámetros:
    - x (numpy.ndarray): Vector de entrada.

    Retorna:
    - float: Valor de la función de Ackley.
    """
    global D
    summation1 = np.sum(np.square(x))
    summation2 = np.sum(np.cos(2.0 * np.pi * x))
    return -20.0 * np.exp(-0.02 * np.sqrt(summation1 / D)) - np.exp(summation2 / D) + 20 + np.e


def ackley_2(x1, x2):
    """
    Función de Ackley 2.

    Parámetros:
    - x1 (float): Primera dimensión.
    - x2 (float): Segunda dimensión.

    Retorna:
    - float: Valor de la función de Ackley.
    """
    return -200 * np.exp(-0.02 * np.sqrt(x1**2 + x2**2))


def ackley_3(x1, x2):
    """
    Función de Ackley 3, una combinación de las funciones de Ackley 1 y 2.

    Parámetros:
    - x1 (float): Primera dimensión.
    - x2 (float): Segunda dimensión.

    Retorna:
    - float: Valor de la función de Ackley.
    """
    return ackley_2(x1, x2) + 5 * np.exp(np.cos(3 * x1) + np.sin(3 * x2))


def ackley_4(x):
    """
    Función de Ackley 4.

    Parámetros:
    - x (numpy.ndarray): Vector de entrada.

    Retorna:
    - float: Valor de la función de Ackley.
    """
    result = 0.0
    for i in range(len(x)-1):
        result += np.exp(-0.2) * np.sqrt(x[i]**2 + x[i+1]**2) + \
            3 * (np.cos(2 * x[i]) + np.sin(2 * x[i+1]))
    return result


def initialize_population(bounds, population_size):
    """
    Inicializa una población aleatoria dentro de los límites dados.

    Parámetros:
    - bounds (numpy.ndarray): Límites para cada dimensión.
    - population_size (int): Tamaño de la población.

    Retorna:
    - list: Lista de individuos (arrays de numpy) representando la población.
    """
    population = []
    for _ in range(population_size):
        individual = np.array([random.uniform(lower, upper)
                              for lower, upper in bounds])
        population.append(individual)
    return population


def evaluate_fitness(fitness_func, population):
    """
    Evalúa la aptitud de cada individuo en la población utilizando la función de aptitud proporcionada.

    Parámetros:
    - fitness_func (function): Función de aptitud.
    - population (list): Lista de individuos.

    Retorna:
    - list: Lista de valores de aptitud correspondientes a cada individuo en la población.
    """
    fitness = []
    for ind in population:
        if fitness_func.__name__ in ['ackley_1', 'ackley_4']:
            fitness.append(fitness_func(np.array(ind)))
        else:
            fitness.append(fitness_func(*ind))
    return fitness


def selection(population, fitness, k=3):
    """
    Selecciona individuos para la reproducción utilizando el método de torneo con k competidores.

    Parámetros:
    - population (list): Lista de individuos.
    - fitness (list): Lista de valores de aptitud correspondientes a cada individuo en la población.
    - k (int): Número de competidores en el torneo.

    Retorna:
    - list: Lista de individuos seleccionados para la reproducción.
    """
    selected = []
    for _ in range(len(population)):
        indices = random.sample(range(len(population)), k)
        candidates = [population[i] for i in indices]
        fitness_values = [fitness[i] for i in indices]
        selected.append(candidates[np.argmin(fitness_values)])
    return selected


def crossover(parents, offspring_size, crossover_rate):
    """
    Realiza el cruce de los padres para producir descendencia.

    Parameters:
    - parents (list): Lista de padres.
    - offspring_size (int): Tamaño de la descendencia.
    - crossover_rate (float): Tasa de cruce.

    Returns:
    - list: Descendencia resultante del cruce.
    """
    offspring = []
    for _ in range(offspring_size):
        parent1, parent2 = random.sample(parents, 2)
        child = []
        for gene1, gene2 in zip(parent1, parent2):
            if random.random() < crossover_rate:
                alpha = random.uniform(0, 1)
                gene = alpha * gene1 + (1 - alpha) * gene2
                child.append(gene)
            else:
                child.append(gene1)
        offspring.append(child)
    return offspring


def mutation(offspring, bounds, mutation_rate):
    """
    Aplica mutaciones a la descendencia.

    Parameters:
    - offspring (list): Descendencia a mutar.
    - bounds (list): Límites para cada dimensión.
    - mutation_rate (float): Tasa de mutación.

    Returns:
    - list: Descendencia mutada.
    """
    mutated_offspring = []
    for individual in offspring:
        mutated_individual = []
        for gene, (lower_bound, upper_bound) in zip(individual, bounds):
            if random.random() < mutation_rate:
                rnd1 = random.random()
                rnd2 = random.uniform(0, 1)
                if rnd1 < 0.5:
                    gene = gene + rnd2 * (upper_bound - gene)
                else:
                    gene = gene - rnd2 * (gene - lower_bound)
            mutated_individual.append(gene)
        mutated_offspring.append(mutated_individual)
    return mutated_offspring


def genetic_algorithm(bounds, fitness_func, population_size, max_generations, crossover_rate, mutation_rate, num_runs):
    """
    Ejecuta un algoritmo genético para optimizar una función de aptitud dada.

    Parámetros:
    - bounds (numpy.ndarray): Límites para cada dimensión de las variables de entrada.
    - fitness_func (function): Función de aptitud a optimizar.
    - population_size (int): Tamaño de la población en cada generación.
    - max_generations (int): Número máximo de generaciones.
    - crossover_rate (float): Tasa de cruce para la reproducción.
    - mutation_rate (float): Tasa de mutación para la descendencia.
    - num_runs (int): Número de ejecuciones independientes del algoritmo genético.

    Retorna:
    - tuple: Una tupla que contiene:
        - best_fitness_values (numpy.ndarray): Matriz de forma (max_generations, num_runs) que almacena los mejores valores de aptitud en cada generación para cada ejecución.
        - results (dict): Un diccionario que contiene listas de tuplas para cada función Ackley. Cada tupla contiene el mejor individuo y su correspondiente valor de aptitud en una ejecución específica.
            Ejemplo de estructura:
            {
                "Ackley 1": [(mejor_individuo, mejor_aptitud), ...],
                "Ackley 2": [(mejor_individuo, mejor_aptitud), ...],
                "Ackley 3": [(mejor_individuo, mejor_aptitud), ...],
                "Ackley 4": [(mejor_individuo, mejor_aptitud), ...],
            }
    """
    best_fitness_values = np.zeros((max_generations, num_runs))
    results = {"Ackley 1": [], "Ackley 2": [], "Ackley 3": [], "Ackley 4": []}

    for run in range(num_runs):
        population = initialize_population(bounds, population_size)

        for i in range(max_generations):
            fitness = evaluate_fitness(fitness_func, population)
            best_fitness_values[i][run] = np.min(fitness)

            parents = selection(population, fitness)
            offspring = crossover(parents, population_size, crossover_rate)
            offspring = mutation(offspring, bounds, mutation_rate)

            population = offspring

            # Guardar el mejor individuo y su fitness para cada función Ackley
            results["Ackley 1"].append(
                (population[np.argmin(fitness)], np.min(fitness)))
            results["Ackley 2"].append(
                (population[np.argmin(fitness)], np.min(fitness)))
            results["Ackley 3"].append(
                (population[np.argmin(fitness)], np.min(fitness)))
            results["Ackley 4"].append(
                (population[np.argmin(fitness)], np.min(fitness)))

    return best_fitness_values, results



# Parámetros del algoritmo genético
population_size = 50
max_generations = 5000
mutation_rate = 0.01
crossover_rate = 0.65
num_seeds = 3


# Definir los límites para cada función
bounds_ackley_1 = [(-32.768, 32.768)] * 10
bounds_ackley_2 = [(-32.768, 32.768)] * 2
bounds_ackley_3 = [(-32.768, 32.768), (-32.768, 32.768)]
bounds_ackley_4 = [(-32.768, 32.768)] * 10

# Llamar a la función del algoritmo genético para cada función
best_fitness_values_1, results_1 = genetic_algorithm(bounds_ackley_1, ackley_1, population_size,
                                                     max_generations,
                                                     crossover_rate,
                                                     mutation_rate,                                                     
                                                     num_seeds)     
best_fitness_values_2, results_2 = genetic_algorithm(bounds_ackley_2, ackley_2, population_size,
                                                     max_generations,
                                                     crossover_rate,
                                                     mutation_rate,                                                     
                                                     num_seeds)
best_fitness_values_3, results_3 = genetic_algorithm(bounds_ackley_3, ackley_3, population_size,
                                                     max_generations,
                                                     crossover_rate,
                                                     mutation_rate,                                                     
                                                     num_seeds)
best_fitness_values_4, results_4 = genetic_algorithm(bounds_ackley_4, ackley_4, population_size,
                                                     max_generations,
                                                     crossover_rate,
                                                     mutation_rate,                        
                                                     num_seeds)


def create_ackley_plots(best_fitness_values, generations, function_name):
    """
    Crea y guarda las figuras de convergencia para la función Ackley.

    Parámetros:
    - best_fitness_values (numpy.ndarray): Matriz de los mejores valores de aptitud para cada ejecución.
    - generations (numpy.ndarray): Arreglo que representa las generaciones.
    - function_name (str): Nombre de la función Ackley.

    Retorna:
    - str: Ruta de la imagen de la figura combinada.
    """
    plt.figure(figsize=(12, 6))
    plt.title(f"{function_name} - Convergence - AG")

    for i in range(best_fitness_values.shape[1]):
        plt.plot(generations, best_fitness_values[:, i], alpha=0.8)

    plt.xlabel('Generations')
    plt.ylabel('Best Fitness Value')
    
    # Crear directorio /Images_AG si no existe
    output_directory = 'Images_AG'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Guardar la figura individual
    individual_figure_path = f"{output_directory}/{function_name}_Convergence.png"
    plt.tight_layout()
    plt.savefig(individual_figure_path)
    plt.close()

    return individual_figure_path

def create_combined_plot(paths, output_filename):
    """
    Combina las figuras individuales en una figura combinada y la guarda.

    Parámetros:
    - paths (list): Lista de rutas de las figuras individuales.
    - output_filename (str): Nombre del archivo de salida para la figura combinada.

    Retorna:
    - str: Ruta de la imagen de la figura combinada.
    """
    plt.figure(figsize=(12, 6))
    plt.suptitle("Combined Convergence Plots")

    for i, path in enumerate(paths, start=1):
        plt.subplot(2, 2, i)
        img = plt.imread(path)
        plt.imshow(img)
        plt.axis('off')

    # Guardar la figura combinada
    output_directory = 'Images_AG'
    combined_figure_path = f"{output_directory}/{output_filename}"
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.savefig(combined_figure_path)
    plt.close()

    return combined_figure_path

generations = np.arange(1, 5001)

# Llamadas a la función para cada función Ackley
ackley_1_path = create_ackley_plots(best_fitness_values_1, generations, "Ackley_1")
ackley_2_path = create_ackley_plots(best_fitness_values_2, generations, "Ackley_2")
ackley_3_path = create_ackley_plots(best_fitness_values_3, generations, "Ackley_3")
ackley_4_path = create_ackley_plots(best_fitness_values_4, generations, "Ackley_4")

# Llamada a la función para la figura combinada
combined_figure_path = create_combined_plot([ackley_1_path, ackley_2_path, ackley_3_path, ackley_4_path], "Combined_Convergence_Plots.png")

# Calcular y mostrar los mejores valores objetivo, mediana y desviación
print("\nMejores valores objetivo:")
print("Ackley 1: {}".format(np.min(best_fitness_values_1[-1, :])))
print("Ackley 2: {}".format(np.min(best_fitness_values_2[-1, :])))
print("Ackley 3: {}".format(np.min(best_fitness_values_3[-1, :])))
print("Ackley 4: {}".format(np.min(best_fitness_values_4[-1, :])))

print("\nMediana:")
print("Ackley 1: {}".format(np.median(best_fitness_values_1[-1, :])))
print("Ackley 2: {}".format(np.median(best_fitness_values_2[-1, :])))
print("Ackley 3: {}".format(np.median(best_fitness_values_3[-1, :])))
print("Ackley 4: {}".format(np.median(best_fitness_values_4[-1, :])))

print("\nDesviación:")
print("Function 1: {}".format(np.std(best_fitness_values_1[-1, :])))
print("Function 2: {}".format(np.std(best_fitness_values_2[-1, :])))
print("Function 3: {}".format(np.std(best_fitness_values_3[-1, :])))
print("Function 4: {}".format(np.std(best_fitness_values_4[-1, :])))
