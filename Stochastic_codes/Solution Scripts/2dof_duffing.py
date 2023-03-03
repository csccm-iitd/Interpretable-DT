# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
x1, x2, x3, x4=0.05,0.05,0,0

T=1
dt=0.001
c1, c2 =4,4
k1, k2 =4000, 2000
alp0 = 10000
sig1, sig2 =0.5, 0.5
m1, m2 =1, 1
t=np.arange(0,T+dt,dt)
nsamp=200
    
y1=[]
y2=[]
y3=[]
y4=[]
nonli1=[]
nonli2=[]
f1=[]
f2=[]
    
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
    


nonli1 = pow(dt,-1)*np.mean(np.array(nonli1), axis = 0)#+np.mean((c1/m1)*y3[:,:-1]+(c2/m1)*(y3[:,:-1]-y4[:,:-1])+(k1/m1)*y1[:,:-1]+(k2/m1)*(y1[:,:-1]-y2[:,:-1]), axis=0)
nonli2 = pow(dt,-1)*np.mean(np.array(nonli2), axis = 0)#+np.mean((c2/m2)*(y4[:,:-1]-y3[:,:-1])+(k2/m2)*(y2[:,:-1]-y1[:,:-1]), axis=0)
f1 = pow(dt,-1)*np.mean(np.array(f1), axis = 0)
f2 = pow(dt,-1)*np.mean(np.array(f2), axis = 0)
    
y1_mean=np.mean(y1,axis=0)
y2_mean=np.mean(y2,axis=0)
y3_mean=np.mean(y3,axis=0)
y4_mean=np.mean(y4,axis=0)
print('xz,xzs')

time = t[0:-1]

print('loop ends here')
   
# %%     
plt.figure(1)
plt.figure(figsize = (10, 10))
plt.subplot(211)
plt.plot(t, y1_mean, 'k')
plt.xlabel('Time in Sec'); 
plt.ylabel('Sample Mean Displacement1');
plt.grid(True); 

plt.subplot(212)
plt.plot(t, y2_mean, 'b')
plt.xlabel('Time in Sec'); 
plt.ylabel('Sample Mean Displacemebt2');
plt.grid(True); 
plt.suptitle('Duffing Van-Der pol')

plt.figure(2)
plt.figure(figsize = (10, 10))
plt.subplot(211)
plt.plot(t, y3_mean, 'k')
plt.xlabel('Time in Sec'); 
plt.ylabel('Sample Mean Velocity1');
plt.grid(True); 

plt.subplot(212)
plt.plot(t, y4_mean, 'b')
plt.xlabel('Time in Sec'); 
plt.ylabel('Sample Mean Velocity2');
plt.grid(True); 
plt.suptitle('Duffing Van-Der pol')
   
# %% 
xdt1=nonli1
xt1=f1
xdt2=nonli2
xt2=f2

plt.figure(5)
plt.plot(y1_mean)
plt.plot(y2_mean)
plt.plot(y3_mean)
plt.plot(y4_mean)

plt.figure(6)
plt.plot(xdt1)
plt.plot(xdt2)

plt.figure(7)
plt.plot(xt1)
plt.plot(xt2)


