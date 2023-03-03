"""
This code belongs to the paper:
-- Probabilistic machine learning based predictive and interpretable digital
    twin for dynamical systems, authored by,
    Tapas Tripura, Aarya Sheetal Desai, Sondipon Adhikari, Souvik Chakraborty
   
*** This code generates the training data and then removes the information of the 
    nominal model.
"""

import numpy as np
from scipy.integrate import solve_ivp
import math


"""
The Response Generation Part: Duffing Oscillator:
"""
def duffing(t0, tn, x0, dt):
    # Function for the dynamical systems:
    def F(t,x, params):
        m, c, k, k3, u = params
        y = np.dot(np.array([[0, 1], [-k/m, -c/m]]), x) \
            + np.dot([0, -k3/m], (x[0]**3)) + np.dot([0, 1/m], u)
        return y
    
    # The time parameters:
    k3 = 100000
    k = 1000
    m = 1
    c = 2
    t_eval = np.arange(0, tn+dt, dt)
    xt = np.vstack(x0)
    uv = []

    # Time integration:
    for i in range(len(t_eval)-1):
        tspan = [t_eval[i], t_eval[i+1]]
        u = np.random.normal(0, 50)
        params = [m, c, k, k3, u]
        uv.append(u)
        
        sol = solve_ivp(F, tspan, x0, method='RK45', t_eval= None, args = (params,))
        solx = np.vstack(sol.y[:,-1])
        xt = np.append(xt, solx, axis=1) # -1 sign is for the last element in array
        x0 = np.ravel(solx)
    uv = np.array(uv)
    # uv = np.append(uv, np.random.normal(0, 50))
    
    # Acceleration vector,
    xdt2 = -(np.multiply(k,xt[0,:-1]) + np.multiply(c,xt[1,:-1]))/m \
                        -(np.multiply(k3, pow(xt[0,:-1],3)))/m + np.array(uv)/m
    
    # Removing the linear structure information,
    xdt = xdt2 +(np.multiply(k,xt[0,:-1]) + np.multiply(c,xt[1,:-1]))/m
    
    xt = xt[:, :-1]
    uv = uv/m
    return xdt, xt, uv, t_eval



"""
The Response Generation Part: 2DOF-Duffing Oscillator:
"""
def twoD_duffing(t0, tn, x0, dt):
    # System parameters:
    c1, c2 = 2, 2
    k1, k2 = 1000, 1000
    alp0 = 100000
    m1, m2 = 1, 1
    
    # The statespace function,
    def F(t, x, params):
        m1, m2, c1, c2, k1, k2, alp0, u1, u2 = params
        y1 = x[2] 
        y2 = x[3]
        y3 = (u1 -(c1*x[2]+c2*(x[2]-x[3])+ \
                k1*x[0] +k2*(x[0]-x[1]) +alp0*pow(x[0],3) +alp0*pow((x[0]-x[1]),3)))/m1    
        y4 = (u2 -(c2*(x[3]-x[2]) +k2*(x[1]-x[0]) +alp0*pow((x[1]-x[0]),3)))/m2
        
        y = [y1, y2, y3, y4]
        return y
    
    t_eval = np.arange(0, tn+dt, dt)
    xt = np.vstack(x0)
    U1, U2 = [], []
    
    # Time integration:
    for i in range(len(t_eval)-1):
        tspan = [t_eval[i], t_eval[i+1]]
        u1, u2 = np.random.normal(0, 50, 2)
        U1.append(u1)
        U2.append(u2)
        params = [m1, m2, c1, c2, k1, k2, alp0, u1, u2]
        sol = solve_ivp(F, tspan, x0, method='RK45', t_eval= None, args=(params,))
        solx = np.vstack(sol.y[:,-1])
        xt = np.append(xt, solx, axis=1) # -1 sign is for the last element in array
        x0 = np.ravel(solx)
   
    U1 = np.append(U1, np.random.normal(0, 50))
    U2 = np.append(U2, np.random.normal(0, 50))
    
    uv = np.vstack(( U1/m1, U2/m2))
    
    # Acceleration vector,
    xdt3 = (U1 - (c1*xt[2,:]+c2*(xt[2,:]-xt[3,:])+k1*xt[0,:]+k2*(xt[0,:]-xt[1,:]) \
                       +alp0*pow(xt[0,:],3)+alp0*pow((xt[0,:]-xt[1,:]),3)))/m1
    xdt4 = (U2 - (c2*(xt[3,:]-xt[2,:])+k2*(xt[1,:]-xt[0,:]) \
                       +alp0*pow((xt[1,:]-xt[0,:]),3)))/m2
    
    # Removing the linear structure information,
    xdt3 = xdt3 +( c1*xt[2,:]+c2*(xt[2,:]-xt[3,:])+k1*xt[0,:]+k2*(xt[0,:]-xt[1,:]) )/m1
    xdt4 = xdt4 +( c2*(xt[3,:]-xt[2,:])+k2*(xt[1,:]-xt[0,:]) )/m2
    
    xdt = np.array([xdt3, xdt4])
    return xdt, xt, uv, t_eval



"""
The Response Generation Part: Crack-propagation:
"""
def crack_degradation(t0, tn, x0, dt):
    # System parameters,
    alp1, alp2, alp3, alp4 = 0.5, 0.5, 1, 1
    gamm = pow(10,-2)
    beta = 2
    m, c, k = 1, 2, 2000
    
    # Statespace,
    def F(t, x, params):
        m, c, k, alp1, alp2, alp3, alp4, gamm, beta, u = params
        y1 = x[1]
        y2 = (u -c*x[1] -(alp1+alp2*math.exp(-alp3*(pow(x[2],alp4))))*k*x[0])/m
        y3 = gamm*pow(x[0]**2 + x[1]**2, beta/2)
        
        y = [y1,y2,y3]
        return y
    
    # Time integration,
    t_eval = np.arange(0, tn+dt, dt)
    xt = np.vstack(x0)
    uv = []
    for i in range(len(t_eval)-1):
        tspan = [t_eval[i], t_eval[i+1]]
        u = np.random.normal(0, 1)
        uv.append(u)
        params = [m, c, k, alp1, alp2, alp3, alp4, gamm, beta, u]
        sol = solve_ivp(F, tspan, x0, method='RK45', t_eval=None, args = (params,))
        solx = np.vstack(sol.y[:,-1])  # -1 sign is for the last element in array
        xt = np.append(xt, solx, axis=1)
        x0 = np.ravel(solx)
        
    uv = np.append(uv, np.random.normal(0, 1))
    
    # The target vectors,
    xdt1 = gamm*pow( xt[0,:]**2 + xt[1,:]**2, beta/2)
    # xdt2 = -(c*xt[1,:] + (alp1+alp2*(np.exp(-alp3*pow(xt[2,:],alp4))))*k*xt[0,:])/m +uv/m
    
    # Removing the linear information,
    # xdt2 = xdt2 + c*xt[1,:]
    xdt2 = -(k/m)*(alp1+alp2*np.exp(-alp3*pow(xt[2,:],alp4)))*xt[0,:] +uv/m
    
    return xdt1, xdt2, xt, uv, t_eval   
    