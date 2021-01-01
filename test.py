import numpy as np
import mpse
import math
import operator
import geppy as gep
def protected_pow(x1, x2):
    # result = np.power(float(abs(x1)),x2)
    try:
        result = abs(x1)**x2
    except:
        result = 2**30
    else:
        result = abs(x1)**x2
    return result

def protected_div(x1, x2):
    if abs(x2) < 1e-6:
        return 1
    return x1 / x2

def testeq(r,tri,dc,ep,ep2,vpm):
    "calc force"
    force = - r*tri/((r - 1*dc)**1)
    # force = (-dc + r + (r + 2)*(tri - 1))/(dc - r)
    # force = tri*(dc*(tri + 2) - dc + r)/(dc - r)
    # r*(r + tri)
    # -5*tri/(2*r)
    # dc*(r**3 + 1) + 2*r*tri
    # dc*tri/r - 2*r + tri
    # -dc + r - 1
    # -dc**2*tri/r
    # dc/r + 2
    # dc*r*tri + dc*r + 2*tri
    # dc + r*tri**2
    # tri*(dc*tri - 5)

    # r*tri/(dc*(dc - r))
    # force = -tri/3 + tri/(dc - r)
    # force = -tri/4 + tri/(dc - r)
    # force = tri*(-dc + r + 1)/(dc - r)
    # (-dc*(dc - r)/5 + tri)/(dc - r)
    # tri*(dc - r + tri - 3)/((dc - r)*(tri - 3))
    # -tri*(dc + 5*r + tri)/((dc + 6*r)*(r - tri))
    # tri*(-dc + r + 1)/(dc - r)
    return force

#input #todo: change ppw to r alpha, use judge fuction to get that can surround. use seed of generation to have same random number in one generation; todo still has 1000 problem
pew = np.array([0,0]) #position of evader in world coordination
# pew = np.array([267.24288514, 223.506317  ]) #position of evader in world coordination
ppw = np.array([[40,30],[5,26],[-35,15],[-60,-30],[-20,-60],[20,-60],[60,-10]]) #position of pursuersin world coordination
ppw = np.array([[60,0],[-30,30*np.sqrt(3)],[-30,-30*np.sqrt(3)]])
# ppw =  np.array([[280.97209982,186.53675321],[231.60216279,236.69141666],[202.59265363,158.60936548]])
vem = 42 #max speed of evader
vpm = 41.5 #max speed of pursuers, here set the same, can change to different
# vpm = float(np.random.uniform(28, 38, 1))
#set
iteration = 1000 #maximun time iteration
func_vec = np.vectorize(testeq)

if 1:
    loop = 1
    escape = 0
    for i in range(loop):
        case = mpse.gen_case(0.8)
        # print(case.vpm)
        it = mpse.get_reward(case,iteration,func_vec,1)
        # print(it)
        if it<iteration:
            escape = escape + 1

    print(escape/loop)