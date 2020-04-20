import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error
from math import sqrt


def generate_data(step=0.02, alpha=-0.1, beta=0.3, gamma=-0.7, delta=0.1):
    x = np.arange(-1, 1, step)
    y = alpha + (beta * x) + (gamma * (x ** 2)) + (delta * (x ** 3))
    return y


def plt_compare(x, y, y_cap, png_name='cmp.png', labels=[]):
    plt.plot(x, y, x, y_cap, label=labels)
    plt.legend(labels)
    plt.savefig(png_name)
    plt.close()


def plot_curve(x, y, png):
    plt.plot(x, y)
    plt.savefig(png)
    plt.close()


def plt_compare2(x, y_act, pop, png):
    plt.plot(x, y_act)
    labels = ['y']
    for idx, i in enumerate(pop):
        y_i = generate_data(alpha=i[1], beta=i[2], gamma=i[3], delta=i[4])
        plt.plot(x, y_i)
        labels.append('y_' + str(idx))
    plt.legend(labels)
    plt.savefig(png)
    plt.close()


def draw_hist(arr, n_bins=100, png="mutation_hist.png", labels=[]):
    plt.hist(arr, bins=n_bins, histtype='bar', edgecolor='black', linewidth=0.4, rwidth=1, label=labels)
    plt.legend()
    plt.savefig(png)
    plt.close()


# Adds Gaussian noise to a parameter
def mutation_genes(init_v=0.1, mu=0, sigma=0.1):
    gn = np.random.normal(mu, sigma, 100)
    arr = np.full(100, init_v)
    arr_pls_gn = np.add(arr, gn)
    return arr_pls_gn

# Generates a population of genes
def produce_pop(pop_size=100, mean=0, sigma=0.1, alpha=-0.1, beta=0.3, gamma=-0.7, delta=0.1):
    alpha_gaus = mutation_genes(alpha, mean, sigma)
    beta_gaus = mutation_genes(beta, mean, sigma)
    gamma_gaus = mutation_genes(gamma, mean, sigma)
    delta_gaus = mutation_genes(delta, mean, sigma)
    pop = np.array([0, 0, 0, 0])
    for i in range(pop_size):
        w = alpha_gaus.item(random.randrange(0, 100, 1))
        x = beta_gaus.item(random.randrange(0, 100, 1))
        y = gamma_gaus.item(random.randrange(0, 100, 1))
        z = delta_gaus.item(random.randrange(0, 100, 1))
        pop = np.vstack((pop, np.array([w, x, y, z])))
    pop = np.delete(pop, 0, 0)
    return pop

# Calculates the loss of all genes and outputs the 10 closest genes to the original
def pop_select(population):
    rmse_arr = []
    for i in range(len(population)):
        y_predicted = generate_data(0.02, alpha=population[i][0], beta=population[i][1], gamma=population[i][2],
                                    delta=population[i][3])
        rms = sqrt(mean_squared_error(y_actual, y_predicted))
        rmse_arr.append([rms, population[i][0], population[i][1], population[i][2], population[i][3]])
    rmse_arr = np.array(rmse_arr)
    rmse_arr.sort(axis=0)
    return rmse_arr[:10]

# Generates a new population of 100 given 10
def evolution_cycle(population):
    rms_e = []
    for i in population:
        pop = pop_select(produce_pop(pop_size=100, mean=0, sigma=0.1, alpha=i[1], beta=i[2], gamma=i[3], delta=i[4]))
        rms_e.extend(pop)
    rms_e = np.array(rms_e)
    rms_e.sort(axis=0)
    return rms_e


y_actual = generate_data(step=0.02, alpha=-0.1, beta=0.3, gamma=-0.7, delta=0.1)
loss = []

# Question 1
y1 = generate_data(step=0.02)
plot_curve(range(0, 100), y1, png="q1.png")

# Question 2
ycap = generate_data(step=0.02, alpha=1, beta=1, gamma=1, delta=1)
plot_curve(range(0, 100), ycap, png="q2.png")
plt_compare(range(0, 100), y1, ycap, png_name='q2_comp.png', labels=['y0', 'y1'])

# Question 3
alpha_mut = mutation_genes(init_v=0)
beta_mut = mutation_genes(init_v=1)
draw_hist(alpha_mut, n_bins=100, labels=['alpha'], png="q3_alpha.png")
draw_hist(beta_mut, n_bins=100, labels=['beta'], png="q3_beta.png")

# Question 4
init_pop = produce_pop(sigma=1.0)  # Generates population of genes of size 100
draw_hist(init_pop[:, (2, 3)], n_bins=100, labels=['beta', 'delta'], png="q4.png")

# Question 5
init_pop = produce_pop()  # Generates population of genes of size 100 (Q5.a)
best_pop = pop_select(population=init_pop)  # Selects the best 10 genes based on their RMSE loss (Q5.b)
pop_mutated = evolution_cycle(
    best_pop)  # Mutates the best 10 genes, generating a new population of the 10 best genes of each evolution (Q5.c)

plt_compare2(range(0, 100), y_actual, best_pop, png='original_population.png')
loss.extend([col[0] for col in best_pop])

# Question 6
old_population = best_pop
for i in range(20):
    new_population = evolution_cycle(old_population[:10])
    plt_compare2(range(0, 100), y_actual, new_population[:10], png='population' + str(i + 1) + '.png')
    loss.extend([col[0] for col in new_population[:10]])
    old_population = new_population
plot_curve(range(len(loss)), loss, 'loss_progression.png')
