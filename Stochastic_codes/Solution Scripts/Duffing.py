# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:10:08 2021

@author: Tapas Tripura
"""

import numpy as np
import matplotlib.pyplot as plt

"""
A linear Dynamical system excited by random noise
The solution is attempted by Taylor 1.5 strong scheme method,
--------------------------------------------------------------------------
"""

# parameters of Duffing oscillator in Equation
# ---------------------------------------------
m = 1
c = 2
k = 1000
k3 = 100000
sigma = 10

# solution by Taylor 1.5 strong scheme Run with dt=0.01
# -------------------------------------------------------
T = 2
dt = 0.001
t = np.arange(0, T+dt, dt)
Nsamp = 100 #int(1/dt) # no. of samples in the run

b = np.array([0, sigma])
delmat = np.row_stack(([np.sqrt(dt), 0],[(dt**1.5)/2, (dt**1.5)/(2*np.sqrt(3))]))

y1 = []
y2 = []
xz = []
xzs = []
# Simulation Starts Here ::
# -------------------------------------------------------
for ensemble in range(Nsamp):
    print(ensemble)
    x0 = np.array([0.1, 0])
    x = np.vstack(x0)  # Zero initial condition.
    for n in range(len(t)-1):
        delgen = np.dot(delmat, np.random.normal(0,1,2))
        dW = delgen[0]
        dZ = delgen[1]
        a1 = x0[1]
        a2 = -(c/m)*x0[1]-(k/m)*x0[0]-(k3/m)*(x0[0]**3)
        b1 = 0
        b2 = (sigma/m)*x0[0]
     #   L0a1 = a2
     #   L0a2 = a1*(-(k/m)-((3*k3/m)*(x0[0]**2)))+a2*(-(c/m))
     #   L0b1 = 0
     #   L0b2 = a1*(sigma/m)
     #   L1a1 = b2
     #   L1a2 = b2*(-(c/m))
     #   L1b1 = 0
     #   L1b2 = 0
        
        # sol1 = x0[0] + a1*dt + 0.5*L0a1*(dt**2) + L1a1*dZ
        # sol2 = x0[1] + a2*dt + b2*dW + 0.5*L0a2*(dt**2) + L1a2*dZ + L0b2*(dW*dt-dZ)
        sol1 = x0[0] + a1*dt 
        sol2 = x0[1] + a2*dt + b2*dW 
        x0 = np.array([sol1, sol2])
        x = np.column_stack((x, x0))
        
    y1.append(x[0,:])
    y2.append(x[1,:])
    
    zint = x[1,0:-1]
    xfinal = x[1,1:] 
    xmz = (xfinal - zint) # 'x(t)-z' vector
    xmz2 = np.multiply(xmz, xmz)
    xz.append(xmz)
    xzs.append(xmz2)
    
    # xz.append(np.mean(xmz))
    # xzs.append(np.mean(np.multiply(xmz, xmz)))
    
xz = pow(dt,-1)*np.mean(np.array(xz), axis = 0)
xzs = pow(dt,-1)*np.mean(np.array(xzs), axis = 0)

y1 = np.array(y1)
y2 = np.array(y2)
u = np.mean(y1, axis = 0)
udot = np.mean(y2, axis = 0)
time = t[0:-1]
# xz = pow(dt,-1)*np.array(xz)
# xzs = pow(dt,-1)*np.array(xzs)

# libr = []
# for j in range(len(y1)):
#     z = np.row_stack((y1[j,:], y2[j,:]))
#     D, nl = fun_library0.library(z, 6, 0)
#     libr.append(D)
# gg= np.mean(np.array(libr), axis = 0)
# D, nl = fun_library0.library(np.row_stack((u, udot)), 6, 0)

plt.figure(1)
plt.figure(figsize = (10, 10))
plt.subplot(211)
plt.plot(t, u, 'k')
plt.xlabel('Time in Sec'); 
plt.ylabel('Sample Mean Displacement');
# plt.grid(True); 

plt.subplot(212)
plt.plot(t, udot, 'b')
plt.xlabel('Time in Sec'); 
plt.ylabel('Sample Mean Velocity');
# plt.grid(True); 
plt.suptitle('Duffing Van-Der pol')

plt.figure(2)
plt.figure(figsize = (10, 8))
plt.subplot(211)
plt.plot(xz, 'k')
plt.xlabel('Time in Sec'); 
plt.ylabel('Absolute Variation');
plt.grid(True);

plt.subplot(212)
plt.plot(xzs, 'k')
plt.xlabel('Time in Sec'); 
plt.ylabel('Quadratic Variation');
plt.grid(True);
    
    