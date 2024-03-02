from matplotlib import pyplot as plt
import numpy as np
import random
import os
import tkinter as tk
from tkinter import messagebox
from pylatex import Document, Section, Subsection,Itemize, Tabular, Center, NoEscape, utils, Package,Command

#===================================================== ACKLEY FUNCTIONS =====================================================

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

#===================================================== Utilities =====================================================

def create_ackley_plots(best_fitness_values, generations, function_name, type):
    """
    Crea y guarda las figuras de convergencia para la función Ackley.

    Parámetros:
    - best_fitness_values (numpy.ndarray): Matriz de los mejores valores de aptitud para cada ejecución.
    - generations (numpy.ndarray): Arreglo que representa las generaciones.
    - function_name (str): Nombre de la función Ackley.
    - type (bool): Boolean que dice si se ejecuta GA (true) o DE (false)

    Retorna:
    - str: Ruta de la imagen de la figura combinada.
    """
    plt.figure(figsize=(12, 6))
    if type:
        plt.title(f"{function_name} - Convergence - AG")
        for i in range(best_fitness_values.shape[1]):
            plt.plot(generations, best_fitness_values[:, i], alpha=0.8)
    else:
        plt.title(f"{function_name} - Convergence - ED")
        for i in range(best_fitness_values.shape[0]):
            plt.plot(generations, best_fitness_values[i, :], alpha=0.8)

    plt.xlabel('Generations')
    plt.ylabel('Best Fitness Value')

    if type:
        output_directory = 'Resultados_AG'
    else:
        output_directory = 'Resultados_ED'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    individual_figure_path = f"{output_directory}/{function_name}_Convergence.png"
    plt.tight_layout()
    plt.savefig(individual_figure_path)
    plt.close()

    return individual_figure_path

def create_combined_plot(paths, output_filename, type):
    """
    Combina las figuras individuales en una figura combinada y la guarda.

    Parámetros:
    - paths (list): Lista de rutas de las figuras individuales.
    - output_filename (str): Nombre del archivo de salida para la figura combinada.
    - type (bool): Boolean que dice si se ejecuta GA (true) o DE (false)

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

    if type:
        output_directory = 'Resultados_AG'
    else:
        output_directory = 'Resultados_ED'
    combined_figure_path = f"{output_directory}/{output_filename}"
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(combined_figure_path)
    plt.show()
    plt.close()

    return combined_figure_path

def generateLatexStatistics(statistics, type):
    """
    Genera un documento LaTeX con estadísticas y gráficos de las ejecuciones del problema.

    Parameters:
    - statistics (list): Lista de estadísticas para diferentes instancias del problema.
    - type (string): 'GA' o 'ED' para multiples fines
    """
    # Crear documento LaTeX
    doc = Document()
    # Añadir el paquetes
    doc.preamble.append(Package('graphicx'))
    doc.preamble.append(Command('usepackage', 'babel', options=['spanish']))
    doc.preamble.append(Package('float'))


    # Crear sección principal
    with doc.create(Section('Estadísticas')):
        for i in range(len(statistics[0])):
            subtitle = 'Resultados ' + statistics[0][i]
            with doc.create(Subsection(subtitle)):
                with doc.create(Itemize()) as itemize:  
                    solution = 'Valor de la función en la mejor solución: ' +  format(statistics[1][i])
                    itemize.add_item(solution)
                    poblation = 'Poblacion de la mejor solución: ' +  format(statistics[2][i])
                    itemize.add_item(poblation)
                    avg = 'Promedio: ' + format(statistics[3][i])
                    itemize.add_item(avg)
                    median = 'Mediana: ' + format(statistics[4][i])
                    itemize.add_item(median)
                    std_dev = 'Desviación Estándar: ' + format(statistics[5][i])
                    itemize.add_item(std_dev)

                    split_parts = statistics[0][i].split(' ')
                    result_string = ' '.join(split_parts[:2])
                    # Agregar la imagen generada al documento
                    img_path = result_string+'_Convergence.png'
                    doc.append(NoEscape(r'\begin{figure}[H]'))
                    doc.append(NoEscape(r'\centering'))
                    doc.append(NoEscape(r'\includegraphics[height=0.5\linewidth]{' + img_path + '}'))
                    doc.append(NoEscape(r'\caption{' + statistics[0][i] +'}'))
                    doc.append(NoEscape(r'\end{figure}'))
        with doc.create(Subsection('Comparativa general')):
            img_path = 'Combined_Convergence_Plots.png'
            doc.append(NoEscape(r'\begin{figure}[H]'))
            doc.append(NoEscape(r'\centering'))
            doc.append(NoEscape(r'\includegraphics[height=0.5\linewidth]{' + img_path + '}'))
            doc.append(NoEscape(r'\caption{Comparativa}'))
            doc.append(NoEscape(r'\end{figure}'))

    # Generar el archivo LaTeX
    doc.generate_tex('Resultados_'+type+'/Estadisticas_'+type)


#===================================================== A.G. =====================================================

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


#===================================================== E.D. =====================================================

def differential_evolution(fitness_func, bounds, population_size, max_generations, F, Cr):
    """
    Implementación del algoritmo de Evolución Diferencial (ED) para optimización.

    Parámetros:
    - fitness_func (function): Función de aptitud que se quiere minimizar.
    - bounds (numpy.ndarray): Límites de las variables de decisión en forma de matriz (shape: (D, 2)).
    - population_size (int): Tamaño de la población.
    - max_generations (int): Número máximo de generaciones.
    - F (float): Factor de escala para la mutación.
    - Cr (float): Tasa de recombinación.

    Retorna:
    - tuple: Tupla que contiene el mejor individuo, su aptitud y una matriz de convergencia.

    Nota:
    - El problema se asume de minimización.
    - La función de aptitud debe ser definida para minimización (retornar un valor menor para mejores soluciones).
    """
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

#===================================================== Parámetros =====================================================
#Generales
max_generations = 5000
population_size = 50

#A.G.
Pm_AG = 0.01
Pc_AG = 0.65

#E.D.
F_ED = 0.5
C_ED = 0.6

#Limites
bounds_1 = np.array([(-35, 35)] * D)
bounds_2 = np.array([(-32, 32), (-32, 32)])
bounds_3 = np.array([(-32, 32), (-32, 32)])
bounds_4 = np.array([(-35, 35)] * 10)

#================================================== Interfaz gráfica ==================================================

def executeAG(num_executions):
    global bounds_1
    global bounds_2
    global bounds_3
    global bounds_4
    global Pm_AG
    global Pc_AG
    best_fitness_values_1, results_1 = genetic_algorithm(bounds_1, ackley_1, population_size,
                                                     max_generations,
                                                     Pc_AG,
                                                     Pm_AG,                                                     
                                                     num_executions)     
    best_fitness_values_2, results_2 = genetic_algorithm(bounds_2, ackley_2, population_size,
                                                        max_generations,
                                                        Pc_AG,
                                                        Pm_AG,                                                     
                                                        num_executions)
    best_fitness_values_3, results_3 = genetic_algorithm(bounds_3, ackley_3, population_size,
                                                        max_generations,
                                                        Pc_AG,
                                                        Pm_AG,                                                     
                                                        num_executions)
    best_fitness_values_4, results_4 = genetic_algorithm(bounds_4, ackley_4, population_size,
                                                        max_generations,
                                                        Pc_AG,
                                                        Pm_AG,                        
                                                        num_executions)
    
    generations = np.arange(1, 5001)

    # Llamadas a la función para cada función Ackley
    ackley_1_path = create_ackley_plots(best_fitness_values_1, generations, "Ackley 1", True)
    ackley_2_path = create_ackley_plots(best_fitness_values_2, generations, "Ackley 2", True)
    ackley_3_path = create_ackley_plots(best_fitness_values_3, generations, "Ackley 3", True)
    ackley_4_path = create_ackley_plots(best_fitness_values_4, generations, "Ackley 4", True)

    # Llamada a la función para la figura combinada
    combined_figure_path = create_combined_plot([ackley_1_path, ackley_2_path, ackley_3_path, ackley_4_path], "Combined_Convergence_Plots.png", True)

    best_inputs_indice_1 = np.argmin(best_fitness_values_1)
    best_inputs_indice_2 = np.argmin(best_fitness_values_2)
    best_inputs_indice_3 = np.argmin(best_fitness_values_3)
    best_inputs_indice_4 = np.argmin(best_fitness_values_4)

    best_inputs_1 = results_1["Ackley 1"][best_inputs_indice_1][0]
    best_inputs_2 = results_2["Ackley 2"][best_inputs_indice_2][0]
    best_inputs_3 = results_3["Ackley 3"][best_inputs_indice_3][0]
    best_inputs_4 = results_4["Ackley 4"][best_inputs_indice_4][0]

    # Mejores valores objetivo
    bf_Ack_1 = np.min(best_fitness_values_1[-1, :])
    bf_Ack_2 = np.min(best_fitness_values_2[-1, :])
    bf_Ack_3 = np.min(best_fitness_values_3[-1, :])
    bf_Ack_4 = np.min(best_fitness_values_4[-1, :])

    # Promedio
    af_Ack_1 = np.average(best_fitness_values_1[-1, :])
    af_Ack_2 = np.average(best_fitness_values_2[-1, :])
    af_Ack_3 = np.average(best_fitness_values_3[-1, :])
    af_Ack_4 = np.average(best_fitness_values_4[-1, :])

    # Mediana
    mf_Ack_1 = np.median(best_fitness_values_1[-1, :])
    mf_Ack_2 = np.median(best_fitness_values_2[-1, :])
    mf_Ack_3 = np.median(best_fitness_values_3[-1, :])
    mf_Ack_4 = np.median(best_fitness_values_4[-1, :])

    # Desviación
    sf_Ack_1 = np.std(best_fitness_values_1[-1, :])
    sf_Ack_2 = np.std(best_fitness_values_2[-1, :])
    sf_Ack_3 = np.std(best_fitness_values_3[-1, :])
    sf_Ack_4 = np.std(best_fitness_values_4[-1, :])

    # Generacion del vector de estadisticas.
    statistics=[]
    statistics.append(['Ackley 1 AG','Ackley 2 AG','Ackley 3 AG','Ackley 4 AG'])
    statistics.append([bf_Ack_1, bf_Ack_2, bf_Ack_3, bf_Ack_4])
    statistics.append([best_inputs_1, best_inputs_2, best_inputs_3, best_inputs_4])
    statistics.append([af_Ack_1, af_Ack_2, af_Ack_3, af_Ack_4])
    statistics.append([mf_Ack_1, mf_Ack_2, mf_Ack_3, mf_Ack_4])
    statistics.append([sf_Ack_1, sf_Ack_2, sf_Ack_3, sf_Ack_4])
    generateLatexStatistics(statistics, 'AG')

def executeED(num_executions):
    global F_ED
    global C_ED

    results = []

    for run in range(num_executions):
        np.random.seed(run)
        best_solution_1, best_fitness_1, convergence_1 = differential_evolution(ackley_1, bounds_1, population_size, max_generations, F_ED, C_ED)
        best_solution_2, best_fitness_2, convergence_2 = differential_evolution(ackley_2, bounds_2, population_size, max_generations, F_ED, C_ED)
        best_solution_3, best_fitness_3, convergence_3 = differential_evolution(ackley_3, bounds_3, population_size, max_generations, F_ED, C_ED)
        best_solution_4, best_fitness_4, convergence_4 = differential_evolution(ackley_4, bounds_4, population_size, max_generations, F_ED, C_ED)

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
    generations = np.arange(1, 5001)

    ackley_1_path = create_ackley_plots(best_fitness_values_ackley_1, generations, "Ackley 1", False)
    ackley_2_path = create_ackley_plots(best_fitness_values_ackley_2, generations, "Ackley 2", False)
    ackley_3_path = create_ackley_plots(best_fitness_values_ackley_3, generations, "Ackley 3", False)
    ackley_4_path = create_ackley_plots(best_fitness_values_ackley_4, generations, "Ackley 4", False)

    combined_figure_path = create_combined_plot([ackley_1_path, ackley_2_path, ackley_3_path, ackley_4_path], "Combined_Convergence_Plots.png", False)

    fitness_values_1 = [result["Ackley 1"][1] for result in results]
    fitness_values_2 = [result["Ackley 2"][1] for result in results]
    fitness_values_3 = [result["Ackley 3"][1] for result in results]
    fitness_values_4 = [result["Ackley 4"][1] for result in results]

    best_inputs_indice_1 = np.argmin(fitness_values_1)
    best_inputs_indice_2 = np.argmin(fitness_values_2)
    best_inputs_indice_3 = np.argmin(fitness_values_3)
    best_inputs_indice_4 = np.argmin(fitness_values_4)

    best_inputs_1 = results[best_inputs_indice_1]["Ackley 1"][0]
    best_inputs_2 = results[best_inputs_indice_2]["Ackley 2"][0]
    best_inputs_3 = results[best_inputs_indice_3]["Ackley 3"][0]
    best_inputs_4 = results[best_inputs_indice_4]["Ackley 4"][0]

    best_fitness_1 = np.min(fitness_values_1)
    best_fitness_2 = np.min(fitness_values_2)
    best_fitness_3 = np.min(fitness_values_3)
    best_fitness_4 = np.min(fitness_values_4)

    average_fitness_1 = np.average(fitness_values_1) 
    average_fitness_2 = np.average(fitness_values_2)
    average_fitness_3 = np.average(fitness_values_3) 
    average_fitness_4 = np.average(fitness_values_4)

    median_fitness_1 = np.median(fitness_values_1) 
    median_fitness_2 = np.median(fitness_values_2)
    median_fitness_3 = np.median(fitness_values_3) 
    median_fitness_4 = np.median(fitness_values_4)

    std_fitness_1 = np.std(fitness_values_1) 
    std_fitness_2 = np.std(fitness_values_2)
    std_fitness_3 = np.std(fitness_values_3) 
    std_fitness_4 = np.std(fitness_values_4)

    # Generacion del vector de estadisticas.
    statistics=[]
    statistics.append(['Ackley 1 ED','Ackley 2 ED','Ackley 3 ED','Ackley 4 ED'])
    statistics.append([best_fitness_1, best_fitness_2, best_fitness_3, best_fitness_4])
    statistics.append([best_inputs_1, best_inputs_2, best_inputs_3, best_inputs_4])
    statistics.append([average_fitness_1, average_fitness_2, average_fitness_3, average_fitness_4])
    statistics.append([median_fitness_1, median_fitness_2, median_fitness_3, median_fitness_4])
    statistics.append([std_fitness_1, std_fitness_2, std_fitness_3, std_fitness_4])
    generateLatexStatistics(statistics, 'ED')

def validate_input(entry):
    try:
        num_executions = int(entry.get())
        if 1 <= num_executions < 30:
            return num_executions
        else:
            raise ValueError("El número de ejecuciones debe estar entre 1 y 29.")
    except ValueError as e:
        messagebox.showerror("Error de entrada", str(e))
        return None

def on_execute_ag(entry, window):
    num_executions = validate_input(entry)
    if num_executions is not None:
        if window:
            window.destroy()
        print("Ejecutando algoritmo....")
        executeAG(num_executions)
        tk.messagebox.showinfo("Finalización","Algoritmo finalizado.\nLos resultados se han almacenado en Resultados_AG.")

def on_execute_ed(entry, window):
    num_executions = validate_input(entry)
    if num_executions is not None:
        if window:
            window.destroy()
        print("Ejecutando algoritmo....")
        executeED(num_executions)
        tk.messagebox.showinfo("Finalización","Algoritmo finalizado.\nLos resultados se han almacenado en Resultados_ED.")

def center_window(window):
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry('{}x{}+{}+{}'.format(width, height, x, y))


def create_window():
    window = tk.Tk()
    window.title("Algoritmos Genéticos y Evolución Diferencial")

    # Etiqueta y entrada para el número de ejecuciones
    label_executions = tk.Label(window, text="Número de ejecuciones:")
    entry_executions = tk.Entry(window, width=5)
    label_executions.grid(row=0, column=0, padx=10, pady=10)
    entry_executions.grid(row=0, column=1, padx=10, pady=10)

    # Botones para ejecutar Algoritmo Genético y Evolución Diferencial
    button_ag = tk.Button(window, text="Ejecutar A.G.", command=lambda: on_execute_ag(entry_executions,window))
    button_ed = tk.Button(window, text="Ejecutar E.D.", command=lambda: on_execute_ed(entry_executions,window))
    button_ag.grid(row=1, column=0, padx=10, pady=10)
    button_ed.grid(row=1, column=1, padx=10, pady=10)

    center_window(window)

    window.mainloop()

# Crear la ventana
create_window()