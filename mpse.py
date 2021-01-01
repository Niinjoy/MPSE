import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

class Case:
   empCount = 0
 
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

def veeq(r,alpha,dc,vem,epsilon,vpm,inputeq):
    "calc ve"
    xforce = inputeq(r,np.cos(alpha),dc,epsilon,down(epsilon),vpm)
    yforce = inputeq(r,np.sin(alpha),dc,epsilon,down(epsilon),vpm)
    xsum = np.sum(xforce) + float(np.random.uniform(1e-10, 1e-8, 1)*np.random.choice([-1,1], 1))
    ysum = np.sum(yforce) + float(np.random.uniform(1e-10, 1e-8, 1)*np.random.choice([-1,1], 1))
    vex = vem*xsum/(np.sqrt(xsum**2+ysum**2))
    vey = vem*ysum/(np.sqrt(xsum**2+ysum**2))
    ve = np.array([vex, vey])
    return ve

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

def gen_case(rate = 1):
    "generate random case, rate 0 to 1, simple to hard"
    #generate range assign by rate, con for constant
    num_min = 3
    num_max_start = 4 # no less than num_min
    num_max_end = 10
    num_max = int(round(weight_boundray(num_max_start, num_max_end, rate)))

    r_min_start = 60
    r_min_end = 20
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
        vpm = np.random.uniform(vpm_min ,vpm_max, num)
        flag = surround_judge(pew,ppw,vem,vpm)
        if flag == 1:
            break
    dc = np.random.uniform(dc_min,dc_max,1) #capture radius
    ti = 0.01 #time interval
    k = 1.9 #parameter in beta equation
    m = 7 #parameter in delta equation
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

    fig = plt.figure()
    fig.canvas.mpl_connect('close_event', handle_close)

    it = 0 #iteration number
    while it < iteration:
        it = it + 1
        if ani != 0:
            # plt.cla()
            plt.axis([-100, 100, -100, 100])
            plt.plot(ppw[:,0],ppw[:,1],marker="o", lw = 0, markersize=4, mec = 'c', mfc = 'g', mew = 0.3)
            plt.plot(pew[0],pew[1], marker="o",  markersize=4, mec = 'b', c = 'k',mew = 0.3)
            plt.pause(0.001)

        danger_num = len(r[r<dc+ti*(vem+vpm)])
        if danger_num !=0 and ani != 0:
            print(danger_num)

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
        if np.min(r)<dc:
            # print('captured!')
            it = iteration*2 - it #make it as reward, the smaller the better
            break
        if (np.max(epsilon_sort)>1.5 and np.min(r)>4*dc) or np.linalg.norm(pew-pew_ini)>r_max_ini*1.5 :
            # print('escaped!')
            break
    if ani != 0:
        # plt.plot(ppw[:,0],ppw[:,1],'or')
        # plt.plot(pew[0],pew[1],'ok')
        plt.show()
    if it == iteration or it < 5:
        print(pew,ppw,epsilon_sort[index_reverse])
    #     it = iteration
    # it = it
    return(it)

#input
pew = np.array([0,0]) #position of evader in world coordination
ppw = np.array([[40,30],[5,26],[-35,15],[-60,-30],[-20,-60],[20,-60],[60,-10]]) #position of pursuersin world coordination
vem = 42 #max speed of evader
vpm = 38 #max speed of pursuers, here set the same, can change to different

#set
dc = 2 #capture radius
ti = 0.01 #time interval
iteration = 1000 #maximun time iteration
k = 1 #parameter in beta equation
m = 6 #parameter in delta equation

# flag = surround_judge(pew,ppw,vem,vpm)
# it = get_reward(pew,ppw,vem,vpm,dc,ti,iteration,k,m,forceeq)
# print(flag)
# print(it)