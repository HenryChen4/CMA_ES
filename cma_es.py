import numpy as np
import matplotlib.pyplot as plt
import mv_helper as mv_helper

def rastrigin(in_matrix):
    A = 10.0
    in_matrix = np.asarray(in_matrix)
    sum_matrix = np.square(in_matrix) - (A*np.cos(2 * np.pi * in_matrix))
    result = np.sum(sum_matrix, axis=1)
    return A*in_matrix.shape[1] + result

def sample_x1_x2(seeds, batch_size, pop_size):
    x_1 = mv_helper.generate_data(seed = seeds[0], m = batch_size)
    x_2 = mv_helper.generate_data(seed = seeds[1], m = batch_size)

    data = np.array([x_1, x_2])

    x_means = mv_helper.get_means(data)
    cov_matrix = mv_helper.est_cov_matrix(data)

    return np.random.multivariate_normal(x_means, cov_matrix, size = pop_size)

def get_best_sols(sample_population, objective_func, percentage):
    fitness = np.reshape(objective_func(sample_population), (objective_func(sample_population).shape[0], 1))
    fitness_sols_map = np.concatenate((fitness, sample_population), axis = 1)

    fitness_sols_map = fitness_sols_map[fitness_sols_map[:, 0].argsort()]
    num_best_samples = int(percentage * fitness_sols_map.shape[0])

    top_sols = np.array(fitness_sols_map[:num_best_samples])
    result_sols = np.delete(top_sols, 0, axis=1)

    return result_sols            

sample_population = sample_x1_x2(seeds=[3, 10], batch_size = 10, pop_size = 5000)
top_solutions = get_best_sols(sample_population = sample_population, objective_func = rastrigin, percentage = 0.25)

plt.plot(sample_population[:, 0], sample_population[:, 1], '.', alpha = 0.5)
plt.plot(top_solutions[:, 0], top_solutions[:, 1], '*', alpha = 0.5) 
plt.axis('equal')
plt.grid()
plt.show()