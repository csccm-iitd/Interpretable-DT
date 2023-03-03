"""
This code belongs to the paper:
-- Probabilistic machine learning based predictive and interpretable digital
    twin for dynamical systems, authored by,
    Tapas Tripura, Aarya Sheetal Desai, Sondipon Adhikari, Souvik Chakraborty
   
*** This code is for the solution of Example 2: two-DOF dynamical system.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

t0=0
tn=2
x0=[0,0.05,0.01,0.005]
t_eval=np.linspace(t0,tn,1001)
   
c1, c2 =2, 2
k1, k2 =1000, 1000
alp0=100000
sig1, sig2 =1, 1
m1, m2 =10, 10
a,b=1,1
    
def F(t,x):
    c1, c2 =2, 2
    k1, k2 =1000, 1000
    alp0=100000
    sig1, sig2 =1, 1
    m1, m2 =10, 10
    a,b=1,1
    
    y1= x[2]
    y2= x[3]
    y3= (sig1*w1+a*(np.sin(2*t))-(c1*x[2]+c2*(x[2]-x[3])+ \
            k1*x[0]+k2*(x[0]-x[1])+alp0*pow(x[0],3)+alp0*pow((x[0]-x[1]),3)))/m1
    y4= (sig2*w2+b*(np.cos(2*t))+np.sin(t)-(c2*(x[3]-x[2])+
    
                                           k2*(x[1]-x[0])+alp0*pow((x[1]-x[0]),3)))/m2
    y=[y1,y2,y3,y4]
    return y
    
dt = 0.001
t_eval = np.arange(0, tn+dt, dt)
xt = np.vstack(x0)
W1, W2=[], []
    # Time integration:
for i in range(len(t_eval)-1):
        tspan = [t_eval[i], t_eval[i+1]]
        w1, w2 = np.random.normal(0, 1, 2)
        W1.append(w1)
        W2.append(w2)
        sol = solve_ivp(F, tspan, x0, method='RK45', t_eval= None)
        solx = np.vstack(sol.y[:,-1])
        xt = np.append(xt, solx, axis=1) # -1 sign is for the last element in array
        x0 = np.ravel(solx)
   
W1 = np.append(W1, np.random.normal(0, 1))
W2 = np.append(W2, np.random.normal(0, 1))
    
f1=a*(np.sin(2*t_eval))
f2=b*(np.cos(2*t_eval))+np.sin(t_eval)
    
uv=np.vstack((f1+sig1*W1,f2+sig2*W2))
    
plt.figure(1)
plt.plot(xt[0,:])
plt.plot(xt[1,:])
plt.plot(xt[2,:])
plt.plot(xt[3,:])
plt.legend(['x1','x2','x3','x4'])