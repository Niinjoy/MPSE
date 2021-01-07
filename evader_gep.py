# pylint: disable=no-member
import geppy as gep
from deap import creator, base, tools
import numpy as np
import random
import operator
import sympy as sp
import math

# for reproduction
s = 0
random.seed(s)
np.random.seed(s)

sym_map = {
    operator.and_.__name__: sp.And,
    operator.or_.__name__: sp.Or,
    operator.not_.__name__: sp.Not,
    operator.add.__name__: operator.add,
    operator.sub.__name__: operator.sub,
    operator.mul.__name__: operator.mul,
    operator.neg.__name__: operator.neg,
    operator.pow.__name__: operator.pow,
    operator.abs.__name__: operator.abs,
    operator.floordiv.__name__: operator.floordiv,
    operator.truediv.__name__: operator.truediv,
    'protected_div': operator.truediv,
    'protected_pow': operator.pow,
    math.log.__name__: sp.log,
    math.sin.__name__: sp.sin,
    math.cos.__name__: sp.cos,
    math.tan.__name__: sp.tan
}

def protected_pow(x1, x2):
    result = np.power(float(abs(x1)),x2)
    # try:
    #     result = abs(x1)**x2
    # except:
    #     result = 2**20
    # else:
    #     result = abs(x1)**x2
    return np.min([result, 2**30])

def protected_div(x1, x2):
    if abs(x2) < 1e-6:
        return 1
    return x1 / x2

import operator 

pset = gep.PrimitiveSet('Main', input_names=['r','tri','dc','ep','ep2','minr','maxr','avgr','stdr'])
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(protected_div, 2)
# pset.add_function(protected_pow, 2)
pset.add_rnc_terminal()

from deap import creator, base, tools

creator.create("FitnessMin", base.Fitness, weights=(-1,))  # to minimize the objective (fitness)
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

h = 8 # head length
n_genes = 2   # number of genes in a chromosome
r = 8   # length of the RNC array

toolbox = gep.Toolbox()
toolbox.register('rnc_gen', random.uniform, a=0, b=5)   # each RNC is random within [0, 5]
toolbox.register('gene_gen', gep.GeneDc, pset=pset, head_length=h, rnc_gen=toolbox.rnc_gen, rnc_array_length=r)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=operator.add)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# compile utility: which translates an individual into an executable function (Lambda)
toolbox.register('compile', gep.compile_, pset=pset)

toolbox.register('select', tools.selTournament, tournsize=3)
# 1. general operators
toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
toolbox.register('mut_invert', gep.invert, pb=0.1)
toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.3)
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)
# 2. Dc-specific operators
toolbox.register('mut_dc', gep.mutate_uniform_dc, ind_pb=0.05, pb=1)
toolbox.register('mut_invert_dc', gep.invert_dc, pb=0.1)
toolbox.register('mut_transpose_dc', gep.transpose_dc, pb=0.1)
# for some uniform mutations, we can also assign the ind_pb a string to indicate our expected number of point mutations in an individual
toolbox.register('mut_rnc_array_dc', gep.mutate_rnc_array_dc, rnc_gen=toolbox.rnc_gen, ind_pb='0.5p')
toolbox.pbs['mut_rnc_array_dc'] = 1  # we can also give the probability via the pbs property

stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

import mpse
iteration = 1000 #maximun time iteration
evtime = 5
# size of population and number of generations
n_pop = 200
n_gen = 200

previous_gen = -1
case_list = [None for _ in range(evtime)]

def evaluate(individual, gen):
    """Evalute the fitness of an individual: MAE (mean absolute error)"""
    global previous_gen, case_list
    func = toolbox.compile(individual)
    func_vec = np.vectorize(func)
    itsum = 0
    for i in range(evtime):
        if gen != previous_gen:
            case_list[i] = mpse.gen_case(1)
            # case_list[i] = mpse.gen_case(gen/n_gen)
            # print('new gen')
        it, danger_num = mpse.get_reward(case_list[i],iteration,func_vec)
        if danger_num == 1:
            it = it + iteration * 2
        if danger_num == 2:
            it = it + iteration * 1
        itsum = itsum + it
    previous_gen = gen
    # print(itsum)
    return itsum/evtime,

toolbox.register('evaluate', evaluate)

pop = toolbox.population(n=n_pop)
hof = tools.HallOfFame(10)   # only record the best 10 individuals ever found in all generations

# start evolution
pop, log = gep.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1, stats=stats, hall_of_fame=hof, verbose=True)

print('Symplified best individual: ')
symplified_best = []
for i in range(len(hof)):
    symplified_best.append(gep.simplify(hof[i], sym_map))
    print(mpse.escape_test(func=np.vectorize(toolbox.compile(hof[i])), loop=1),'  ', symplified_best[i])

for i in symplified_best:
    print(i)
