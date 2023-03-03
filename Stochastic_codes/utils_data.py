"""
This code belongs to the paper:
-- Probabilistic machine learning based predictive and interpretable digital
    twin for dynamical systems, authored by,
    Tapas Tripura, Aarya Sheetal Desai, Sondipon Adhikari, Souvik Chakraborty
   
*** This code generates the training data and then removes the information of the 
    nominal model.
"""

import numpy as np
import matplotlib.pyplot as plt

"""
A Duffing system excited by random noise
----------------------------------------------------------------------
"""
def duffing(x1, x2, T, dt):
    # parameters of Duffing oscillator in Equation
    m = 1
    c = 2
    k = 1000
    k3 = 100000
    sigma = 0.5
    
    # solution by Taylor 1.5 strong scheme Run with dt=0.01
    # -------------------------------------------------------
    t = np.arange(0, T+dt, dt)
    Nsamp = 100 # no. of samples in the run
    
    y1, y2, xz, xzs = [], [], [], []
    # Simulation Starts Here ::
    # -------------------------------------------------------
    for ensemble in range(Nsamp):
        x0 = np.array([x1, x2])
        x = np.vstack(x0) # Zero initial condition.
        for n in range(len(t)-1):
            dW = np.sqrt(dt)*np.random.randn(1)
            a1 = x0[1]
            a2 = -(c/m)*x0[1]-(k/m)*x0[0]-(k3/m)*(x0[0]**3)
            b1 = 0
            b2 = (sigma/m)

            sol1 = x0[0] + a1*dt 
            sol2 = x0[1] + a2*dt + b2*dW 
            x0 = np.hstack([sol1, sol2])
            x = np.column_stack((x, x0))
           
        y1.append(x[0,:])
        y2.append(x[1,:])
        
        zint = x[1,0:-1]
        xfinal = x[1,1:] 
        xmz = (xfinal - zint) # 'x(t)-z' vector
        xmz2 = np.multiply(xmz, xmz)
        xz.append(xmz)
        xzs.append(xmz2)
            
    y1 = np.array(y1)
    y2 = np.array(y2)

    xz = pow(dt,-1)*np.mean(np.array(xz), axis = 0)+np.mean(c*y2[:,:-1]+k*y1[:,:-1], axis=0)
    xzs = pow(dt,-1)*np.mean(np.array(xzs), axis = 0)
    time = t[0:-1]
    
    return xz, xzs, y1, y2, time



"""
A 2-DOF Duffing system excited by random noise
----------------------------------------------------------------------
"""
def twoDOF_duffing(x1, x2, x3, x4, T, dt):
    # parameters of Duffing oscillator in Equation
    c1, c2 = 4, 4
    k1, k2 = 4000, 2000
    alp0 = 50000
    sig1, sig2 = 0.5, 0.5
    m1, m2 = 1, 1
    t = np.arange(0,T+dt,dt)
    nsamp = 200
    y1, y2, y3, y4 = [], [], [], []
    nonli1, nonli2, f1, f2 = [], [], [], []
        
    for ensemble in range(nsamp):
        x0 = np.array([x1, x2, x3, x4])
        x = np.vstack(x0) # Zero initial condition.
        for n in range(len(t)-1):
          dW = np.sqrt(dt)*np.random.randn(2)
                
          a1 = x0[2]
          a2 = x0[3]
          a3 = -(c1/m1)*x0[2]-(c2/m1)*(x0[2]-x0[3])-(k1/m1)*x0[0]-(k2/m1)*(x0[0]-x0[1])-(alp0/m1)*(x0[0]**3)-(alp0/m1)*((x0[0]-x0[1])**3)
          a4 = -(c2/m2)*(x0[3]-x0[2])-(k2/m2)*(x0[1]-x0[0])-(alp0/m2)*((x0[1]-x0[0])**3)
          b1 = sig1/m1
          b2 = sig2/m2

          sol1 = x0[0] + a1*dt 
          sol2 = x0[1] + a2*dt
          sol3 = x0[2] + a3*dt + b1*dW[0]
          sol4 = x0[3] + a4*dt + b2*dW[1] 
          x0 = np.hstack([sol1, sol2, sol3, sol4])
          x = np.column_stack((x, x0))
                
        y1.append(x[0,:])
        y2.append(x[1,:])
        y3.append(x[2,:])
        y4.append(x[3,:])
            
        zint1 = x[2,0:-1]
        xfinal1 = x[2,1:] 
            
        zint2 = x[3,0:-1]
        xfinal2 = x[3,1:]
            
        xz1 = (xfinal1 - zint1) # 'x(t)-z' vector
        xmz1 = np.multiply(xz1, xz1)
            
        xz2 = (xfinal2 - zint2) # 'x(t)-z' vector
        xmz2 = np.multiply(xz2, xz2)
            
        nonli1.append(xz1)
        nonli2.append(xz2)
        f1.append(xmz1)
        f2.append(xmz2)
                
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    y4 = np.array(y4)
        
    nonli1 = pow(dt,-1)*np.mean(np.array(nonli1), axis = 0)+np.mean((c1/m1)*y3[:,:-1]+(c2/m1)*(y3[:,:-1]-y4[:,:-1])+(k1/m1)*y1[:,:-1]+(k2/m1)*(y1[:,:-1]-y2[:,:-1]), axis=0)
    nonli2 = pow(dt,-1)*np.mean(np.array(nonli2), axis = 0)+np.mean((c2/m2)*(y4[:,:-1]-y3[:,:-1])+(k2/m2)*(y2[:,:-1]-y1[:,:-1]), axis=0)
    f1 = pow(dt,-1)*np.mean(np.array(f1), axis = 0)
    f2 = pow(dt,-1)*np.mean(np.array(f2), axis = 0)
    time = t[0:-1]

    return nonli1, f1, nonli2, f2, y1, y2, y3, y4, time
        


"""
A Linear degrading system excited by random noise
----------------------------------------------------------------------
"""
def Crack_degradation(x1, x2, x3, T, dt):
    # parameters of Duffing oscillator in Equation
    alp1, alp2, alp3, alp4 = 0.5, 0.5, 1, 1
    gamm = pow(10,-2)
    beta = 2
    m, c, k = 1, 2, 2000
    sig1 = 1
    t = np.arange(0,T+dt,dt)
    Nsamp = 200
    y1, y2, y3 = [], [], []
    nonli1, f1, degrade = [], [], []
    
    for ensemble in range(Nsamp):
        x0 = np.array([x1, x2, x3])
        x = np.vstack(x0) # Zero initial condition.
        for n in range(len(t)-1):
          dW = np.sqrt(dt)*np.random.randn(1)
          a1 = x0[1]
          a2 = -(c/m)*x0[1]-(k/m)*alp1*x0[0]-(k/m)*alp2*np.exp(-alp3*pow(x0[2], alp4))*x0[0]
          a3 = gamm*pow((x0[0]**2+x0[1]**2), beta/2)
          b1 = sig1/m
          
          sol1 = x0[0]+a1*dt
          sol2 = x0[1]+a2*dt+b1*dW
          sol3 = x0[2]+a3*dt
          
          x0 = np.hstack([sol1, sol2, sol3])
          x = np.column_stack((x, x0))
          
        y1.append(x[0,:])
        y2.append(x[1,:])
        y3.append(x[2,:])
              
        zint = x[2,0:-1]
        xfinal = x[2,1:] 
              
        zint1 = x[1,0:-1]
        xfinal1 = x[1,1:]
              
        xz1 = (xfinal1 - zint1) # 'x(t)-z' vector
        xmz1 = np.multiply(xz1, xz1)
        xz = (xfinal - zint) # 'x(t)-z' vector
        
        nonli1.append(xz)
        degrade.append(xz1)
        f1.append(xmz1)
                  
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    
    nonli1 = pow(dt,-1)*np.mean(np.array(nonli1), axis = 0)
    degrade = pow(dt,-1)*np.mean(np.array(degrade),axis=0)+np.mean(np.array((c/m)*y2[:,:-1]),axis=0)
    f1 = pow(dt,-1)*np.mean(np.array(f1), axis = 0)
    time = t[0:-1]

    return nonli1, degrade, f1, y1, y2, y3, time
    