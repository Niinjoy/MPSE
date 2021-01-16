import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import multiprocessing
from functools import partial

def sin(x):
    return np.sin(x)
def cos(x):
    return np.cos(x)
def Abs(x):
    return abs(x)

class EvLambda( object ):
    def __init__( self, body ):
        self.body= body
    def __call__( self, r,tri,dc,ep,ep2,vpm,vem,minr,maxr,avgr,stdr):
        return eval( self.body )
    def __str__( self ):
        return self.body

def veeq(r,alpha,dc,vem,epsilon,vpm,forceeq):
    "calc ve"
    force = np.zeros((2,r.shape[0]))
    force_sum = np.zeros(2)
    ep2 = down(epsilon)
    minr = np.min(r)
    maxr = np.max(r)
    avgr = np.mean(r)
    stdr = np.std(r)
    for i in range(2):
        tri = np.cos(alpha) if i==0 else np.sin(alpha)
        force[i] = forceeq(r, tri, dc, epsilon, ep2, vpm, vem, minr, maxr, avgr, stdr)
    force_sum = np.sum(force, axis=1) + np.random.uniform(1e-10, 1e-8, 2) * np.random.choice([-1,1], 2)
    ve = vem*force_sum/np.linalg.norm(force_sum)
    return ve

class PuLambda( object ):
    def __init__( self, body ):
        self.body= body
    def __call__( self, adep, r, rater, dc,ep,ep2,vpm,vem,minr,maxr,avgr,stdr):
        return eval( self.body )
    def __str__( self ):
        return self.body

def vpeq(r, alpha, theta, epsilon, vpm, m, k, dc, vem, pu_lambda):
    "calc vp from begining"
    if pu_lambda == 0:
        delta = deltaeq(epsilon, down(epsilon), up(theta), down(theta), m)
        gamma = gammaeq(r, down(r), up(r))
        beta = betaeq(delta, gamma, k)
    else:
        beta = pu_lambda(abs(epsilon-down(epsilon)), r, r/((down(r)+r+up(r))/3), dc, epsilon, down(epsilon), vpm, vem, np.min(r), np.max(r), np.mean(r), np.std(r))
    sd = np.sign(epsilon - down(epsilon))
    vp = vpeq_beta(alpha, sd, vpm, beta)
    return vp

def_ev_lambda = EvLambda('-r*tri/(r-dc)')
def_pu_lambda = PuLambda('0') #('dep + dth + r + avg3r')
colors=['b','c','g','m','r','y','g','m','r','y'] #color of pursuer

class Case: 
   def __init__(self, pew, ppw, vem, vpm, dc, ti, k, m):
      self.pew = pew
      self.ppw = ppw
      self.vem = vem
      self.vpm = vpm
      self.dc = dc
      self.ti = ti
      self.k = k
      self.m = m
   
def polecoor(x, y):
    "cart to pole coordinate"
    r = np.sqrt(x**2 + y**2)
    alpha = np.arctan2(y, x)
    # alpha[alpha<0] = alpha[alpha<0]+2*np.pi #the same effect
    alpha = np.where(alpha > 0, alpha, alpha + 2*np.pi)
    return(alpha, r)

def pol2cart(alpha, r):
    "pole to cart coordinate"
    x = r * np.cos(alpha)
    y = r * np.sin(alpha)
    return(x, y)

def up(i):
    "roll ndarray to the left"
    ii = np.roll(i,-1)
    return(ii)

def down(i):
    "roll ndarray to the right"
    ii = np.roll(i,1)
    return(ii)

def weight_boundray(start,end,rate):
    "get weighted boundray by start, end, and rate"
    boundray = np.average((start,end),weights=[1-rate,rate])
    return boundray

def epsiloneq(alphaupi,alphai,thetaupi,thetai):
    "clac epsilon"
    epsilon = np.where(alphaupi > alphai, 0, 2*np.pi) + alphaupi-alphai-(thetaupi+thetai)/2
    return epsilon

def deltaeq(epsilon_i,epsilon_downi,theta_upi,theta_downi,m):
    "clac delta"
    delta = (2*np.abs(epsilon_i-epsilon_downi)/(4*np.pi-theta_upi+theta_downi))**(1/m)
    return delta

def gammaeq(r_i,r_downi,r_upi):
    "clac gamma"
    gamma = np.sin(np.pi*(r_i/(r_i+r_downi+r_upi))**(np.log(2)/np.log(3)))
    return gamma

def betaeq(delta,gamma,k):
    "clac beta"
    beta = np.pi/2*(1-np.exp(-k*delta*gamma))
    return beta

def vpeq_beta(alpha,sd,vpm,beta):
    "calc vp using beta"
    vs = vpm*np.sin(beta)
    vh = vpm*np.cos(beta)
    a_s = alpha+sd*np.pi/2
    a_h = alpha+np.pi
    x=vs*np.cos(a_s)+vh*np.cos(a_h)
    y=vs*np.sin(a_s)+vh*np.sin(a_h)
    vp = np.vstack((x,y)).T
    return vp

# def forceeq(r,tri,dc):
#     "calc force"
#     force = - r*tri/((r - 1*dc)**0.9)
#     return force

def surround_judge(pew,ppw,vem,vpm):
    "judge whether surrounded in the initial state"
    num = ppw.shape[0] #number of pursuers
    vpm = vpm * np.ones(num)
    lam = vpm/vem
    theta = 2*np.arcsin(lam)
    alpha = polecoor(ppw[:,0] - pew[0], ppw[:,1] - pew[1])[0]
    epsilon = epsiloneq(up(alpha), alpha, up(theta), theta)
    # thetaG = np.sum(theta) + np.sum(epsilon[epsilon<0])
    if np.max(epsilon)>=0:
        flag = 0
    else:
        flag = 1
    return flag

def surround_judge_alpha(alpha, theta):
    "judge whether surrounded in the initial state"
    epsilon = epsiloneq(up(alpha), alpha, up(theta), theta)
    if np.max(epsilon)>=0:
        flag = 0
    else:
        flag = 1
    return flag

def gen_case(rate = 1, k = 1.9, m = 7):
    "generate random case, rate 0 to 1, simple to hard"
    #generate range assign by rate, con for constant
    num_min = 3
    num_max_start = 4 # no less than num_min
    num_max_end = 10
    num_max = int(round(weight_boundray(num_max_start, num_max_end, rate)))

    r_min_start = 60
    r_min_end = 40
    r_min = weight_boundray(r_min_start, r_min_end, rate)
    r_max = 80

    vem_value = 42
    
    vpm_min = 32
    vpm_max_start = 33
    vpm_max_end = 38
    vpm_max = weight_boundray(vpm_max_start, vpm_max_end, rate)

    dc_min_start = 2
    dc_min_end = 1
    dc_min = weight_boundray(dc_min_start, dc_min_end, rate)
    dc_max_start = dc_min_start
    dc_max_end = 4
    dc_max = weight_boundray(dc_max_start, dc_max_end, rate)

    vem = vem_value
    num = np.random.choice(np.arange(num_min, num_max+1), 1)
    if num == 3: # special deal for num=3, vpm should biger than 36.4
        vpm_limit = vem*np.sin(np.pi/num)+0.5
        vpm_min = max(vpm_min,vpm_limit)
        vpm_max = max(vpm_min+0.5, vpm_max)
    vpm = np.random.uniform(vpm_min, vpm_max, 1) # 1 for the same, num for different
    # print(num, vpm)
    vpm = vpm * np.ones(num)
    theta = 2*np.arcsin(vpm/vem)
    while 1:
        alpha = np.sort(np.random.uniform(0, 2*np.pi, num))
        flag = surround_judge_alpha(alpha, theta)
        if flag == 1:
            break
    r = np.random.uniform(r_min, r_max, num)
    dc = np.random.uniform(dc_min,dc_max,1) #capture radius
    ti = 0.01 #time interval
    if k == 0:
        k = np.random.uniform(1,2,1) #1.9 #parameter in beta equation
    if m == 0:
        m = np.random.uniform(3,7,1) #7 #parameter in delta equation
    x = r * np.cos(alpha)
    y = r * np.sin(alpha)
    ppw = np.vstack((x,y)).T
    pew = np.array([0,0]) 
    case = Case(pew, ppw, vem, vpm, dc, ti, k, m)
    return case

def handle_close(evt):
    print('Closed Figure!')
    sys.exit()

def get_reward(case,iteration,inputeq,ani = 0,pursuer=0):
    "get reward, pursuer=-1 for danger_num go back"
    pew = case.pew
    ppw = case.ppw
    vem = case.vem
    vpm = case.vpm
    dc = case.dc
    ti = case.ti
    k = case.k
    m = case.m

    num = ppw.shape[0] #number of pursuers
    theta = 2*np.arcsin(vpm/vem)
    [alpha,r] = polecoor(ppw[:,0] - pew[0], ppw[:,1] - pew[1])
    index = np.argsort(alpha)
    index_reverse = np.argsort(index)
    alpha_sort = alpha[index]
    r_sort = r[index]
    vpm_sort = vpm[index]
    theta_sort = theta[index]
    epsilon_sort = epsiloneq(up(alpha_sort), alpha_sort, up(theta_sort), theta_sort)
    r_max_ini = np.max(r)
    pew_ini = pew[:]
    danger_dis = dc+ti*(vem+vpm)

    if pursuer <= 0: #test evader
        pu_lambda = 0
        ev_lambda = inputeq
    else: #test pursuer
        pu_lambda = inputeq
        ev_lambda = def_ev_lambda

    if ani != 0:
        fig = plt.figure()
        fig.canvas.mpl_connect('close_event', handle_close)
        plt.axis('equal')
        plt.axis([-135, 135, -100, 100])
        plt.scatter(ppw[:,0], ppw[:,1], marker = "^", s = 15, c = colors[0:num], alpha = 1)
        plt.scatter(pew[0],pew[1], marker="x",  s=15, c = 'k', alpha = 1)
        plt.pause(0.001)

    it = 0 #iteration number
    while it < iteration:
        it = it + 1

        #pursuer strategy
        vp_sort = vpeq(r_sort, alpha_sort, theta_sort, epsilon_sort, vpm_sort, m, k, dc, vem, pu_lambda)
        vp = vp_sort[index_reverse]

        danger_num = len(r[r<danger_dis])
        if danger_num !=0:
            if ani != 0:
                print(danger_num)

        #evader strategy
        ev_lambda_use = ev_lambda
        ve = veeq(r_sort,alpha_sort,dc,vem,epsilon_sort,vpm_sort,ev_lambda_use)
        if pursuer == -1:#use danger_num go back
            if danger_num == 1:
                ev_lambda_use = EvLambda('-(r+0)*tri/(r-dc)**1.1')
                ve = veeq(r_sort,alpha_sort,dc,vem,epsilon_sort,vpm_sort,ev_lambda_use)
            if danger_num == 2:
                # ev_lambda_use = EvLambda('-(r+0)*tri/(r-dc)**1.2')
                # ve = veeq(r_sort,alpha_sort,dc,vem,epsilon_sort,vpm_sort,ev_lambda_use)
                danger_alpha = alpha_sort[np.argwhere(r_sort<danger_dis)].flatten()#two danger alpha
                alpha_e = np.mean(danger_alpha) - (danger_alpha[1]-danger_alpha[0] < np.pi)*np.pi# the back direction
                ve = vem*np.array([np.cos(alpha_e),np.sin(alpha_e)])
            
        #update position
        ppw = ppw + vp*ti
        pew = pew + ve*ti
        [alpha,r] = polecoor(ppw[:,0] - pew[0], ppw[:,1] - pew[1])
        index = np.argsort(alpha)
        index_reverse = np.argsort(index)
        alpha_sort = alpha[index]
        r_sort = r[index]
        vpm_sort = vpm[index]
        theta_sort = theta[index]
        epsilon_sort = epsiloneq(up(alpha_sort), alpha_sort, up(theta_sort), theta_sort)
        if ani != 0:
            # plt.cla()
            plt.axis('equal')
            plt.axis([-135, 135, -100, 100])
            plt.scatter(ppw[:,0], ppw[:,1], marker = "o", s = 8, c = colors[0:num], alpha = 0.2)
            plt.scatter(pew[0],pew[1], marker="o",  s=8, c = 'k', alpha = 0.2)
            plt.pause(0.001)

        if np.min(r)<dc:
            if ani != 0:
                print('captured!')
            it = iteration*2 - it #make it as reward, the smaller the better
            break
        if np.min(r)>3*dc and (np.max(epsilon_sort)>1.2 or np.linalg.norm(pew-pew_ini)>r_max_ini*1.5) :
            if ani != 0:
                print('escaped!')
            break
    # if it == iteration or it < 5:
    #     if pursuer == 0:
    #         print(pew,ppw,epsilon_sort[index_reverse])

    final_danger_num = len(r[r<danger_dis*1.7])
    if final_danger_num != 0:
        if ani != 0:
            print('final:',final_danger_num)
        if it<iteration:#sometimes final_danger_dis > escape_dis
            final_danger_num = 0
            # print(danger_dis*1.7, 3*dc, np.sort(r), pew, ppw)
        # print(final_danger_num, np.sort(r)[final_danger_num:]/danger_dis[0])
    if ani != 0:
        plt.show()
        plt.close()
    return(it, final_danger_num)

def capture_test(func = def_ev_lambda, loop = 1000, rate = 1, iteration = 1000, k = 1.9, m = 7, pursuer = 0):
    "test the capture rate"
    # func_vec = np.vectorize(func)
    ani = 0
    if loop == 0: #run one time with animation
        loop = 1
        ani = 1
    # start = time.time()
    case_list = [gen_case(rate, k, m) for _ in range(loop)]
    # end = time.time()
    # print("case, generated, time spent: {} s".format(round(end - start,2)))
    get_reward_case = partial(get_reward, iteration=iteration,inputeq=func,ani=ani,pursuer=pursuer)
    myPool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool_tuple = myPool.map(get_reward_case, case_list)
    myPool.close()
    myPool.join()
    it_list = np.array([i[0] for i in pool_tuple])
    danger_num_list = np.array([i[1] for i in pool_tuple])
    capture = len(it_list[it_list>iteration])
    danger_num_1 = len(danger_num_list[danger_num_list==1])
    danger_num_2 = len(danger_num_list[danger_num_list==2])
    return(capture/loop, [danger_num_1/loop, danger_num_2/loop])

def get_ev_lambda_list(*param):
    "get a list of evader lambda function from str list"
    lambda_list = []
    for eq in param:
        lambda_list.append(EvLambda(eq))
    return lambda_list

def get_pu_lambda_list(*param):
    "get a list of pursuer lambda function from str list"
    lambda_list = []
    for eq in param:
        lambda_list.append(PuLambda(eq))
    return lambda_list

def get_list_capture_rate(lambda_list, loop = 1000):
    "get the capture rate of a list of lambda function"
    for i in range(len(lambda_list)):
        print(loop, 'times', capture_test(lambda_list[i], loop), '  ', lambda_list[i])

if __name__ == '__main__':
    # np.random.seed(0)
    start = time.time()
    good_ev_lambda_list = get_ev_lambda_list('-r*tri/(r-dc)',
    ' tri*(4.66*dc - vem*(minr - (vpm*(ep2 + r))**(2*dc/vpm)))/vem ',
    ' tri*(1.47*dc - vem*(minr - (vpm*(ep2 + r))**(2*dc/vpm)))/vem ',
    ' tri*(-avgr*(minr - (vpm*(ep2 + r))**(2*dc/vpm)) + 1.47*dc)/avgr ',
	'r*tri*vpm*(dc*ep + vpm)/(dc - r)',
    'ep + r*tri*vpm**2/(dc - r) + tri/(r*vpm)',
    'r*tri*vpm*(vpm + 0.9)/(dc - r)',
    'tri*(0.2*dc**2*tri*(dc - r) + ep2*r*vpm**2)/(ep2*(dc - r))',
    'tri*(dc*(dc - r)*(dc - tri) + ep2*r**2*vpm**2)/(ep2*r*(dc - r))',
    '(ep**2*(dc - r)*(dc*ep2 - r) + r**2*tri*vpm**2)/(r*(dc - r))',
    'tri*(dc + r*vpm**2 - r)/(dc - r)',
    'tri*(r + 1)/(dc - r)',
    '-tri/3 + tri/(dc - r)',
    '(dc*r**3*tri + (2.8 - 1.4*tri)*(dc - r))/(dc*r**2*(dc - r))',
    'r*(dc - r + stdr*tri*vem)/(dc - r)',
    '(avgr*r*tri*vem*(tri - vpm) + 1.36*maxr*stdr*(dc - r))/((dc - r)*(tri - vpm))',
    'r*(dc + maxr*tri*vem - r)/(dc - r)',
    '(maxr*r*tri*vem + minr*(dc - r))/(dc - r)',
    ) # capture rate < 0.7
    good_pu_lambda_list = get_pu_lambda_list('adep*minr/(dc*vem) + sin(adep)','0')
    # ev_lambda_list = get_ev_lambda_list(    )
    # get_list_capture_rate(ev_lambda_list, 1000)
    ev_lambda_test = EvLambda('-(r+0)*tri/(r-dc)')
    print(capture_test(func=ev_lambda_test, loop=0, pursuer=-1),ev_lambda_test)
    # print(capture_test(func=good_pu_lambda_list[0], loop=1000, pursuer=1))
    end = time.time()
    print("time spent: {} s".format(round(end - start,2)))