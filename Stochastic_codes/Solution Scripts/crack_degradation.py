# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:02:51 2022

@author: Lenovo
"""
import numpy as np
import matplotlib.pyplot as plt
T=1
x1,x2,x3=0.5,0.05,0
alp1,alp2,alp3,alp4=0.5,0.5,1,1
gamm=pow(10,-2)
beta=2
m,c,k=1,2,2000
sig1=1
dt=0.001
t=np.arange(0,T+dt,dt)
Nsamp=200
y1,y2,y3=[],[],[]
nonli1,f1,degrade=[],[],[]
for ensemble in range(Nsamp):
    x0 = np.array([x1, x2, x3])
    x = np.vstack(x0) # Zero initial condition.
    for n in range(len(t)-1):
                
          dW = np.sqrt(dt)*np.random.randn(1)
          a1=x0[1]
          a2=-(c/m)*x0[1]-(k/m)*(alp1+alp2*(np.exp(-alp3*pow(x0[2],alp4))))*x0[0]
          a3=gamm*pow((x0[0]**2+x0[1]**2),beta/2)
          b1=sig1/m
          
          sol1=x0[0]+a1*dt
          sol2=x0[1]+a2*dt+b1*dW
          sol3=x0[2]+a3*dt
          
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

          
y1_mean=np.mean(y1,axis=0)
y2_mean=np.mean(y2,axis=0)
y3_mean=np.mean(y3,axis=0)
          
print('xz,xzs')
time = t[0:-1]
print('loop ends here')
                    
plt.figure(1)
plt.figure(figsize = (10, 10))
plt.subplot(211)
plt.plot(t, y1_mean, 'k')
plt.xlabel('Time in Sec'); 
                
plt.subplot(212)
plt.plot(t, y2_mean, 'b')
plt.xlabel('Time in Sec'); 
plt.ylabel('Sample Mean Displacement');
plt.grid(True); 
plt.suptitle('Crack degradation')
        
plt.figure(2)
plt.figure(figsize = (10, 10))
plt.plot(t, y3_mean, 'k')
plt.xlabel('Time in Sec'); 
plt.ylabel('Sample Mean Velocity');
plt.grid(True); 
        
plt.figure(3)
plt.figure(figsize = (10, 8))
plt.subplot(211)
plt.plot(xz, 'k')
plt.xlabel('Time in Sec'); 
plt.ylabel('Non linear function');
plt.grid(True);
        
plt.subplot(212)
plt.plot(xmz1, 'k')
plt.xlabel('Time in Sec'); 
plt.ylabel('Force');
plt.grid(True);
        
            
xdt1=nonli1
xdt2=degrade
xt1=f1
degradation_term = np.array(np.exp(-alp3*pow(y3_mean,alp4))*y1_mean)
        
        
        
plt.figure(5)
plt.plot(y1_mean)
plt.plot(y2_mean)
plt.plot(y3_mean)
plt.legend(('y1_mean','y2_mean','y3_mean'))
        
plt.figure(6)
plt.plot(xdt1)
plt.ylabel('Nonlinear function')
plt.xlabel('time')
        
plt.figure(7)
plt.plot(xt1)
plt.ylabel('Unknown forcing function')
plt.xlabel('time')
    
plt.figure(8)
plt.plot(xdt2)
plt.ylabel('degrade')
plt.xlabel('time')
