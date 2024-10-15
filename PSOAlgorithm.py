import random
import numpy as np
from math import floor

def MM_generate_miss_prpo(metabolites, data, mis_prop, alpha, beta, gamma):
    """
    mis_prop: missing rate
    alpha: Separation of low concentrations from the rest of the metabolites (proportion of low concentrations)
    beta: Separate high concentrations from the rest of the metabolites (medium to low concentration ratio)
    gamma: The loss rate of low concentration as a percentage of the total loss rate
    """

    total_num = data.size
    low_mnar_percentage = gamma * mis_prop
    mid_mnar_percentage = 0.5 * gamma * mis_prop

    low_abundance_missing = round(low_mnar_percentage * total_num)
    mid_abundance_missing = round(mid_mnar_percentage * total_num)
    mv_num = round(mis_prop * total_num) - low_abundance_missing - mid_abundance_missing
    if mv_num < 0:
        return []

    data = data.values.astype(float)
    mean_concentrations = np.mean(data, axis=1)
    metabolites = metabolites.tolist()
    sorted_metabolites = [x for _, x in sorted(zip(mean_concentrations, metabolites))]

    low_abundance_num = round(alpha * data.shape[0])
    mid_abundance_num = round(beta * data.shape[0])
    low_abundance_metabolites = sorted_metabolites[:low_abundance_num]
    mid_abundance_metabolites = sorted_metabolites[low_abundance_num:mid_abundance_num]

    if len(low_abundance_metabolites)==0 or len(mid_abundance_metabolites)==0:
        return []
    tmp_num = low_abundance_missing % len(low_abundance_metabolites)
    for metabolite in low_abundance_metabolites:
        num_missing = floor(low_abundance_missing / len(low_abundance_metabolites))
        if tmp_num > 0:
            num_missing += 1
            tmp_num -= 1
        if num_missing >= len(data[metabolites.index(metabolite)]):
            return []
        low_indices_missing = np.argpartition(data[metabolites.index(metabolite)], num_missing - 1)[:round(num_missing * 0.8)]
        all_indices_missing = np.argpartition(data[metabolites.index(metabolite)], num_missing - 1)[:num_missing]
        kk = metabolites.index(metabolite)
        data[kk][low_indices_missing] = np.nan
        tmp = np.setdiff1d(all_indices_missing, low_indices_missing)
        if floor(num_missing * 0.2) > 0:
            tmp = list(tmp)
            low_indices_missing = random.sample(tmp, floor(num_missing * 0.2))
            data[kk][low_indices_missing] = np.nan

    tmp_num = mid_abundance_missing % len(mid_abundance_metabolites)
    for metabolite in mid_abundance_metabolites:
        num_missing = floor(mid_abundance_missing / len(mid_abundance_metabolites))
        if tmp_num > 0:
            num_missing += 1
            tmp_num -= 1
        if num_missing >= len(data[metabolites.index(metabolite)]):
            return []
        mid_indices_missing = np.argpartition(data[metabolites.index(metabolite)], num_missing - 1)[
                              :round(num_missing * 0.8)]
        all_indices_missing = np.argpartition(data[metabolites.index(metabolite)], num_missing - 1)[:num_missing]
        kk = metabolites.index(metabolite)
        data[kk][mid_indices_missing] = np.nan
        tmp = np.setdiff1d(all_indices_missing, mid_indices_missing)
        if floor(num_missing * 0.2) > 0:
            tmp = list(tmp)
            mid_indices_missing = random.sample(tmp, floor(num_missing * 0.2))
            data[kk][mid_indices_missing] = np.nan

    non_nan_coordinates = np.column_stack(np.where(~np.isnan(data)))
    non_nan_flat = [item[0] * data.shape[1] + item[1] for item in non_nan_coordinates]
    mcar_missing = random.sample(non_nan_flat, mv_num)
    data.flat[mcar_missing] = np.nan

    data_miss_prpo = []
    for item in data:
        data_miss_prpo.append(np.count_nonzero(np.isnan(item)) / len(item))
    return data_miss_prpo


def evaluate(individual, metabolites, data, mis_prop, orl_miss_prpo, item_k):
    x, y, z, gamma = individual
    constraint = x + y + z - 1
    if constraint != 0:
        return np.inf
    if z <= mis_prop*gamma or y<= mis_prop*gamma*0.5:
        return np.inf
    euclidean_distance = 0
    for i in range(item_k):
        tmp_miss_prpo = MM_generate_miss_prpo(metabolites, data, mis_prop, z, y+z,gamma)
        if len(tmp_miss_prpo) != len(orl_miss_prpo):
            return np.inf
        euclidean_distance += np.linalg.norm(np.array(orl_miss_prpo) - np.array(tmp_miss_prpo))
    return euclidean_distance / item_k


def initialize_particles(n_particles, n_dimensions, bounds, mis_prop):
    particles = []
    for _ in range(n_particles):
        valid_particle = False
        while not valid_particle:
            particle = np.random.uniform(bounds[:, 0], bounds[:, 1], (1, n_dimensions))[0]
            sum_x = particle[0] + particle[1]
            if 1 - sum_x > 0:
                particle[2] = 1 - sum_x
                if particle[2]*particle[3]*0.9+particle[1]*particle[3]*0.9 < mis_prop:
                    valid_particle = True
        particles.append(particle)
    return np.array(particles)


def pso_search(objective_func, n_particles, n_dimensions, max_iterations, bounds,
               metabolites, data, mis_prop, orl_miss_prpo, item_k):
    particles = initialize_particles(n_particles, n_dimensions, bounds, mis_prop)
    velocities = np.zeros((n_particles, n_dimensions))

    best_positions = particles.copy()
    best_fitnesses = np.array([objective_func(individual, metabolites, data, mis_prop, orl_miss_prpo, item_k)
                               for individual in particles])

    global_best_index = np.argmin(best_fitnesses)
    global_best_position = best_positions[global_best_index].copy()
    global_best_fitness = best_fitnesses[global_best_index]

    # 迭代更新
    for iteration in range(max_iterations):
        for i in range(n_particles):
            velocities[i] = velocities[i] + random.random() * (best_positions[i] - particles[i]) + random.random() * \
                            (global_best_position - particles[i])
            particles[i] = particles[i] + velocities[i]

            particles[i] = np.clip(particles[i], bounds[:, 0], bounds[:, 1])

            fitness = objective_func(particles[i], metabolites, data, mis_prop, orl_miss_prpo, item_k)

            if fitness < best_fitnesses[i]:
                best_positions[i] = particles[i].copy()
                best_fitnesses[i] = fitness

                if fitness < global_best_fitness:
                    global_best_position = particles[i].copy()
                    global_best_fitness = fitness
    return global_best_position, global_best_fitness


def pso_xyz_gradient(metabolites, data, orl_data, mis_prop, item_k):
    orl_miss_prpo = []
    orl_data = orl_data.values.astype("float")
    for item in orl_data:
        orl_miss_prpo.append(np.count_nonzero(np.isnan(item)) / len(item))

    n_particles = 100
    n_dimensions = 4
    max_iterations = 300
    bounds = [[0, 1], [0, 1], [0, 1], [0, 0.8]]
    bounds = np.array(bounds)
    best_position, best_fitness = pso_search(evaluate, n_particles, n_dimensions, max_iterations, bounds,
                                             metabolites, data, mis_prop, orl_miss_prpo, item_k)
    x, y, z, tmp_mis = best_position
    alpha = z
    beta = y+z
    gamma = tmp_mis
    return alpha, beta, gamma
