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



def differential_evolution(fitness_func, bounds, population_size, max_generations, F=0.8, Cr=0.6):
    D = len(bounds)
    population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(population_size, D))
    fitness = np.zeros(population_size)
    convergence = np.zeros(max_generations)

    for i in range(population_size):
        if fitness_func.__name__ in ['ackley_1', 'ackley_4']:
            fitness[i] = fitness_func(population[i])
        else:
            fitness[i] = fitness_func(*population[i])

    best_fitness = np.min(fitness)
    best_solution = population[np.argmin(fitness)]

    for generation in range(max_generations):
        convergence[generation] = best_fitness

        for i in range(population_size):
            indices = [idx for idx in range(population_size) if idx != i]
            a, b, c = np.random.choice(indices, size=3, replace=False)
            mutant = population[a] + F * (population[b] - population[c])
            mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])

            trial = np.copy(population[i])
            j_rand = np.random.randint(D)

            for j in range(D):
                if np.random.rand() < Cr or j == j_rand:
                    trial[j] = mutant[j]

            if fitness_func.__name__ in ['ackley_1', 'ackley_4']:
                trial_fitness = fitness_func(trial)
            else:
                trial_fitness = fitness_func(*trial)

            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness

            if trial_fitness < best_fitness:
                best_fitness = trial_fitness
                best_solution = trial

    return best_solution, best_fitness, convergence

# ================================ EJECUCION ================================
population_size = 50
max_generations = 5000
F = 0.5
Cr = 0.6
num_runs = 25

bounds_1 = np.array([(-35, 35)] * 10)  # D=10
bounds_2 = np.array([(-32, 32), (-32, 32)])
bounds_3 = np.array([(-32, 32), (-32, 32)])
bounds_4 = np.array([(-35, 35)] * 10)

results = []

for run in range(num_runs):
    np.random.seed(run)
    best_solution_1, best_fitness_1, convergence_1 = differential_evolution(ackley_1, bounds_1, population_size, max_generations, F, Cr)
    best_solution_2, best_fitness_2, convergence_2 = differential_evolution(ackley_2, bounds_2, population_size, max_generations, F, Cr)
    best_solution_3, best_fitness_3, convergence_3 = differential_evolution(ackley_3, bounds_3, population_size, max_generations, F, Cr)
    best_solution_4, best_fitness_4, convergence_4 = differential_evolution(ackley_4, bounds_4, population_size, max_generations, F, Cr)

    results.append({
        "Ackley 1": (best_solution_1, best_fitness_1, convergence_1),
        "Ackley 2": (best_solution_2, best_fitness_2, convergence_2),
        "Ackley 3": (best_solution_3, best_fitness_3, convergence_3),
        "Ackley 4": (best_solution_4, best_fitness_4, convergence_4)
    })
    
best_fitness_values_ackley_1 = np.array([result["Ackley 1"][2] for result in results])
best_fitness_values_ackley_2 = np.array([result["Ackley 2"][2] for result in results])
best_fitness_values_ackley_3 = np.array([result["Ackley 3"][2] for result in results])
best_fitness_values_ackley_4 = np.array([result["Ackley 4"][2] for result in results])

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
    plt.title(f"{function_name} - Convergence - ED")

    for i in range(best_fitness_values.shape[0]):
        plt.plot(generations, best_fitness_values[i, :], alpha=0.8)

    plt.xlabel('Generations')
    plt.ylabel('Best Fitness Value')
    
    # Crear directorio /Images_ED si no existe
    output_directory = 'Images_ED'
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
    output_directory = 'Images_ED'
    combined_figure_path = f"{output_directory}/{output_filename}"
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.savefig(combined_figure_path)
    plt.close()

    return combined_figure_path

generations = np.arange(1, 5001)



# Llamadas a la función para cada función Ackley 
ackley_1_path = create_ackley_plots(best_fitness_values_ackley_1, generations, "Ackley_1")
ackley_2_path = create_ackley_plots(best_fitness_values_ackley_2, generations, "Ackley_2")
ackley_3_path = create_ackley_plots(best_fitness_values_ackley_3, generations, "Ackley_3")
ackley_4_path = create_ackley_plots(best_fitness_values_ackley_4, generations, "Ackley_4")

# Llamada a la función para la figura combinada
combined_figure_path = create_combined_plot([ackley_1_path, ackley_2_path, ackley_3_path, ackley_4_path], "Combined_Convergence_Plots.png")

fitness_values_1 = [result["Ackley 1"][1] for result in results]
fitness_values_2 = [result["Ackley 2"][1] for result in results]
fitness_values_3 = [result["Ackley 3"][1] for result in results]
fitness_values_4 = [result["Ackley 4"][1] for result in results]

best_fitness_1 = np.min(fitness_values_1)
best_fitness_2 = np.min(fitness_values_2)
best_fitness_3 = np.min(fitness_values_3)
best_fitness_4 = np.min(fitness_values_4)

median_fitness_1 = np.median(fitness_values_1) 
median_fitness_2 = np.median(fitness_values_2)
median_fitness_3 = np.median(fitness_values_3) 
median_fitness_4 = np.median(fitness_values_4)

std_fitness_1 = np.std(fitness_values_1) 
std_fitness_2 = np.std(fitness_values_2)
std_fitness_3 = np.std(fitness_values_3) 
std_fitness_4 = np.std(fitness_values_4)

print("Best fitness values: ")
print("Ackley 1: ", best_fitness_1)
print("Ackley 2: ", best_fitness_2)
print("Ackley 3: ", best_fitness_3)
print("Ackley 4: ", best_fitness_4)
print()
print("Median fitness values:")
print("Ackley 1: ", median_fitness_1)
print("Ackley 2: ", median_fitness_2)
print("Ackley 3: ", median_fitness_3)
print("Ackley 4: ", median_fitness_4)
print()
print("Standard deviation of fitness values: ")
print("Ackley 1: ", std_fitness_1)
print("Ackley 2: ", std_fitness_2)
print("Ackley 3: ", std_fitness_3)
print("Ackley 4: ", std_fitness_4)

print('--------------------')
print(fitness_values_1)
print(fitness_values_2)
print(fitness_values_3)
print(fitness_values_4)