# pylint: disable=no-member
import geppy as gep
from deap import creator, base, tools
import numpy as np
import random
import operator
import sympy as sp
import math
import multigep
import time
import pathos.multiprocessing as mp

# for reproduction
# s = 0
# random.seed(s)
# np.random.seed(s)

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
    result = np.power(np.max([float(abs(x1)), 1e-6]),np.median([x2,20,-20]))
    # if math.isinf(result) == True:
    #     print(x1,x2,'inf!')
    return np.min([result, 2**20])

def protected_div(x1, x2):
    if abs(x2) < 1e-6:
        return 1
    return x1 / x2

import operator 

pset = gep.PrimitiveSet('Main', input_names=['adep', 'r', 'rater','dc','ep','ep2','vpm','vem','minr','maxr','avgr','stdr'])
pset.add_rnc_terminal()
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(protected_div, 2)
pset.add_function(protected_pow, 2)
# pset.add_function(operator.abs, 1)
pset.add_function(math.sin, 1)
# pset.add_function(math.cos, 1)
pset.add_constant_terminal(np.pi)
pset.add_constant_terminal(np.e)

from deap import creator, base, tools

creator.create("FitnessMin", base.Fitness, weights=(1,))  # to minimize the objective (fitness)
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

h = 10 # head length
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
toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=2 / (2 * h + 1), pb=1)
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
dev = 0
if dev == 0:
    evtime = 10
    n_pop = 1000
    n_gen = 400
    loop = 1000
    print("Using", format(mp.cpu_count()), "CPUs, estimated time", round(evtime*n_gen*n_pop/mp.cpu_count()*0.273558/60/60, 2), "h")
else: # develop mode
    evtime = 3
    n_pop = 5
    n_gen = 5
    loop = 10
rate = 1
curriculum_learning = 0
curriculum_ratio = 0.75
curriculum_gen = int(round(curriculum_ratio * n_gen)) # The gens used for learning. The gens after will use rate = 1

previous_gen = -1
start = time.time()
if curriculum_learning == 0:
    case_list = [mpse.gen_case(rate) for _ in range(evtime*(n_gen+1))]
else:
    case_list = [mpse.gen_case(i//evtime/curriculum_gen) for i in range(evtime*(n_gen+1))]

# case_num = 200
# case_list = [mpse.gen_case(rate) for _ in range(case_num)]
# index = np.random.randint(0,case_num,evtime*(n_gen+1))

end = time.time()
print("Time spent for case generation: {} s".format(round(end - start,2)))
# print("Time spent for case generation: {} s.".format(round(end - start,2)),"Using {} CPUs.".format(mp.cpu_count()))

def evaluate(ind_and_gen):
    """Evalute the fitness of an individual: MAE (mean absolute error)"""
    individual = ind_and_gen[0]
    gen = ind_and_gen[1]
    func = toolbox.compile(individual)
    func_vec = np.vectorize(func)
    itsum = 0
    for i in range(evtime):
        it= mpse.get_reward(case_list[i+gen*evtime],iteration,func_vec,0,1)[0]
        # it= mpse.get_reward(case_list[index[i+gen*evtime]],iteration,func_vec,0,1)[0]
        # if danger_num == 1:
        #     it = it + iteration * 2
        # if danger_num == 2:
        #     it = it + iteration * 1
        itsum = itsum + it
    return itsum/evtime,

toolbox.register('evaluate', evaluate)

pop = toolbox.population(n=n_pop)
hof = tools.HallOfFame(10)   # only record the best 10 individuals ever found in all generations

# start evolution
start = time.time()
pop, log = multigep.gep_multi(pop, toolbox, n_generations=n_gen, n_elites=3, stats=stats, hall_of_fame=hof, verbose=True)
end = time.time()
time_spent = round(end - start,2)
print("time spent: {} s".format(time_spent), "= {} min".format(round(time_spent/60,2)), "= {} h".format(round(time_spent/3600,2)))

print('\nSymplified best individual: ')
symplified_best_list = []
result_list = []
for i in range(len(hof)):
    symplified_best = gep.simplify(hof[i], sym_map)
    if symplified_best not in symplified_best_list:
        symplified_best_list.append(symplified_best)
        result = mpse.capture_test(func=np.vectorize(toolbox.compile(hof[i])), loop=loop, pursuer=1)
        print(result,'  ', symplified_best)
        result_list.append(result)

print('\n', len(symplified_best_list), 'different items')
for i in range(len(symplified_best_list)):
    print("\'" + str(symplified_best_list[i]) + "\', #" + str(result_list[i]))