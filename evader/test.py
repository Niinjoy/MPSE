import numpy as np
import mpse
import geppy as gep

def testeq(r,tri,dc,ep,ep2,vpm):
    "calc force"
    force = - r*tri/((r - 1*dc)**1)
    return force

# testeq0 = lambda r,tri,dc,ep,ep2: - r*tri/((r - dc))

# print(mpse.escape_test(func = testeq0, loop = 0))