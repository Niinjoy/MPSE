import numpy as np
import matplotlib.pyplot as plt
import sys

class EvLambda( object ):
    def __init__( self, body ):
        self.body= body
    def __call__( self, r,tri,dc,ep,ep2,minr,maxr,avgr,stdr):
        return eval( self.body )
    def __str__( self ):
        return self.body

def veeq(r,alpha,dc,vem,epsilon,vpm,inputeq):
    "calc ve"
    force = np.zeros((2,r.shape[0]))
    force_sum = np.zeros(2)
    for i in range(2):
        tri = np.cos(alpha) if i==0 else np.sin(alpha)
        force[i] = inputeq(r, tri, dc, epsilon, down(epsilon), np.min(r), np.max(r), np.mean(r), np.std(r))
    force_sum = np.sum(force, axis=1) + np.random.uniform(1e-10, 1e-8, 2) * np.random.choice([-1,1], 2)
    ve = vem*force_sum/np.linalg.norm(force_sum)
    return ve

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

def vpeq(r, alpha, theta, epsilon, vpm, m, k):
    "calc vp from begining"
    delta = deltaeq(epsilon, down(epsilon), up(theta), down(theta), m)
    gamma = gammaeq(r, down(r), up(r))
    sd = np.sign(epsilon - down(epsilon))
    beta = betaeq(delta, gamma, k)
    vp = vpeq_beta(alpha, sd, vpm, beta)
    return vp

def forceeq(r,tri,dc):
    "calc force"
    force = - r*tri/((r - 1*dc)**0.9)
    return force

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

    while 1:
        num = np.random.choice(np.arange(num_min, num_max+1), 1)
        alpha = np.sort(np.random.uniform(0, 2*np.pi, num))
        r = np.random.uniform(r_min, r_max, num)
        x = r * np.cos(alpha)
        y = r * np.sin(alpha)
        ppw = np.vstack((x,y)).T
        pew = np.array([0,0]) 
        vem = vem_value
        vpm = np.random.uniform(vpm_min ,vpm_max, 1) # 1 for the same, num for different
        flag = surround_judge(pew,ppw,vem,vpm)
        if flag == 1:
            break
    dc = np.random.uniform(dc_min,dc_max,1) #capture radius
    ti = 0.01 #time interval
    if k == 0:
        k = np.random.uniform(1,2,1) #1.9 #parameter in beta equation
    if m == 0:
        m = np.random.uniform(3,7,1) #7 #parameter in delta equation
    case = Case(pew, ppw, vem, vpm, dc, ti, k, m)
    return case

def handle_close(evt):
    print('Closed Figure!')
    sys.exit()

def get_reward(case,iteration,inputeq,ani = 0):
    "get reward"
    pew = case.pew
    ppw = case.ppw
    vem = case.vem
    vpm = case.vpm
    dc = case.dc
    ti = case.ti
    k = case.k
    m = case.m

    num = ppw.shape[0] #number of pursuers
    vpm = vpm * np.ones(num)
    lam = vpm/vem
    theta = 2*np.arcsin(lam)
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
        vp_sort = vpeq(r_sort, alpha_sort, theta_sort, epsilon_sort, vpm_sort, m, k)
        vp = vp_sort[index_reverse]

        #evader strategy
        ve = veeq(r_sort,alpha_sort,dc,vem,epsilon_sort,vpm_sort,inputeq)

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
        danger_num = len(r[r<danger_dis])
        if danger_num !=0 and ani != 0:
            print(danger_num)
        if ani != 0:
            # plt.cla()
            plt.axis('equal')
            plt.axis([-135, 135, -100, 100])
            plt.scatter(ppw[:,0], ppw[:,1], marker = "o", s = 8, c = colors[0:num], alpha = 0.2)
            plt.scatter(pew[0],pew[1], marker="o",  s=8, c = 'k', alpha = 0.2)
            plt.pause(0.001)

        if np.min(r)<dc:
            # print('captured!')
            it = iteration*2 - it #make it as reward, the smaller the better
            break
        if (np.max(epsilon_sort)>1.2 and np.min(r)>3*dc) or np.linalg.norm(pew-pew_ini)>r_max_ini*1.5 :
            # print('escaped!')
            break
    if it == iteration or it < 5:
        print(pew,ppw,epsilon_sort[index_reverse])

    if danger_num != 0 and ani != 0:
        print(danger_num)
    if ani != 0:
        plt.show()
    return(it, danger_num)

# testeq0 = lambda r,tri,dc,ep,ep2: -r*tri/((r-dc))

def escape_test(func, loop = 1000, rate = 1, iteration = 1000, k = 1.9, m = 7):
    "test the escape rate"
    # func_vec = np.vectorize(func)
    escape = 0
    danger_num_1 = 0 # number of danger_num == 1
    danger_num_2 = 0 # number of danger_num == 2
    ani = 0
    if loop == 0: #run one time with animation
        loop = 1
        ani = 1
    for _ in range(loop):
        case = gen_case(rate, k, m)
        it, danger_num = get_reward(case,iteration,func,ani)
        if it<iteration:
            escape = escape + 1
        if danger_num == 1:
            danger_num_1 = danger_num_1 + 1
        if danger_num == 2:
            danger_num_2 = danger_num_2 + 1
    return(escape/loop, [danger_num_1/loop, danger_num_2/loop])

def get_lambda_list(*param):
    "get a list of lambda function from str list"
    lambda_list = []
    for eq in param:
        lambda_list.append(EvLambda(eq))
    return lambda_list

def get_list_escape_rate(lambda_list, loop = 1000):
    "get the escape rate of a list of lambda function"
    for i in range(len(lambda_list)):
        print(loop, 'times', escape_test(lambda_list[i], loop)[0], '  ', lambda_list[i])

if __name__ == '__main__':
    lambda_list = get_lambda_list('-r*tri/(r-dc)',
    'tri*(r + 1)/(dc - r)',
    '-tri/3 + tri/(dc - r)',
    '(dc*r**3*tri + (2.8 - 1.4*tri)*(dc - r))/(dc*r**2*(dc - r))',
    )
    # get_list_escape_rate(lambda_list, 1000)
    print(escape_test(func=lambda_list[0], loop=1))