"""
This code belongs to the paper:
-- Probabilistic machine learning based predictive and interpretable digital
    twin for dynamical systems, authored by,
    Tapas Tripura, Aarya Sheetal Desai, Sondipon Adhikari, Souvik Chakraborty
   
*** This code is for the Fig10. Long-term prediction of example-2.
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 26


"""
The Response Generation Part: 2DOF-Duffing Oscillator:
"""
def twoD_duffing(t0, tn, x0, dt, sys, force):
    # System parameters:
    m1, m2, c1, c2, k1, k2, alp0 =sys
    force1, force2 = force[0], force[1]

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
    # Time integration:
    for i in range(len(t_eval)-1):
        tspan = [t_eval[i], t_eval[i+1]]
        u1, u2 = force1[i], force2[i]
        params = [m1, m2, c1, c2, k1, k2, alp0, u1, u2]
        sol = solve_ivp(F, tspan, x0, method='RK45', t_eval= None, args=(params,))
        solx = np.vstack(sol.y[:,-1])
        xt = np.append(xt, solx, axis=1) # -1 sign is for the last element in array
        x0 = np.ravel(solx)
        
    return xt


# %%
np.random.seed(0)

# System parameters:
c1, c2 = 2, 2
k1, k2 = 1000, 1000
alp0 = 100000
m1, m2 = 1, 1

sys = [m1, m2, c1, c2, k1, k2, alp0]

# The time parameters:
x0 = np.array([0, 0, 0, 0])
t0, tn, dt = 0, 500, 0.01
t = np.arange(t0, tn+dt, dt)

# Response generation:
force = np.random.normal(0, 50, [2, len(t)-1])

xt = twoD_duffing(t0, tn, x0, dt, sys, force=force)

# %%
np.random.seed(0)

xt_i = []
# System parameters:
for i in range(100):
    print(i)
    
    c1, c2 = 2, 2
    k1, k2 = 1000, 1000
    alp0 = np.random.normal(100000,1.43)
    m1, m2 = 1, 1
    
    sys = [m1, m2, c1, c2, k1, k2, alp0]

    # Response generation:
    xt_temp = twoD_duffing(t0, tn, x0, dt, sys, force=force)
    xt_i.append(xt_temp)

xt_i = np.array(xt_i)

# %%
xt_mean = np.mean(xt_i, axis=0)
xt_u = xt_mean + 2*np.std(xt_i, axis=0)
xt_l = xt_mean - 2*np.std(xt_i, axis=0)

# %%
"""
Broken Axis plot ...
"""
figure2, ax = plt.subplots(2,3, figsize=(22, 12), sharey='row', facecolor='w')
figure2.subplots_adjust(wspace=0.05, hspace=0.25)  # adjust space between axes
x1range = np.array([51, 301, 491])*100; x2range = np.array([52, 302, 492])*100;

# plot the same data on both axes
for i in range(len(ax)*len(ax[0])):
    if i<3:
        ax[0,i].plot(t[x1range[i]:x2range[i]], xt[2, x1range[i]:x2range[i]], 'r')
        ax[0,i].plot(t[x1range[i]:x2range[i]], xt_mean[2, x1range[i]:x2range[i]], 'b:', linewidth=4)
        ax[0,i].fill_between(t[x1range[i]:x2range[i]],  xt_u[2, x1range[i]:x2range[i]],
                             xt_l[2, x1range[i]:x2range[i]], alpha = 1, color = 'darkorchid')
        # ax[0,i].set_ylim(-5, 5)
        ax[0,i].tick_params(axis='x', labelrotation=45)
        ax[0,i].ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
        ax[0,i].grid(True)
    else:
        ax[1,i-3].plot(t[x1range[i-3]:x2range[i-3]], xt[3, x1range[i-3]:x2range[i-3]], 'r')
        ax[1,i-3].plot(t[x1range[i-3]:x2range[i-3]], xt_mean[3, x1range[i-3]:x2range[i-3]], 'b:', linewidth=4)
        ax[1,i-3].fill_between(t[x1range[i-3]:x2range[i-3]],  xt_u[3, x1range[i-3]:x2range[i-3]],
                             xt_l[3, x1range[i-3]:x2range[i-3]], alpha = 1, color = 'darkorchid')
        # ax[1,i-3].set_ylim(-5, 5)
        ax[1,i-3].tick_params(axis='x', labelrotation=45)
        ax[1,i-3].ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
        if i == 4:
            ax[1,i-3].set_xlabel(' Time (s)', fontweight='bold');
        ax[1,i-3].grid(True)
        
# hide the spines between ax and ax2
ax[0,0].spines['right'].set_visible(False)
ax[1,0].spines['right'].set_visible(False)
ax[0,-1].spines['left'].set_visible(False)
ax[1,-1].spines['left'].set_visible(False)
ax[0,0].yaxis.tick_left(); ax[1,0].yaxis.tick_left()
ax[0,-1].yaxis.tick_right(); ax[1,-1].yaxis.tick_right()
ax[0,-1].yaxis.set_ticks_position('right'); ax[1,-1].yaxis.set_ticks_position('right')
ax[0,0].set_ylabel(' 'r'${{\rm E}[X]}$', fontweight='bold')
ax[1,0].set_ylabel(' 'r'${{\rm E}[\dot{X}]}$', fontweight='bold');
ax[0,1].legend(['Actual system','Identified system'], ncol=2, fontsize=30)

for i in range(1, len(ax[0])-1):
    ax[0,i].spines['right'].set_visible(False)
    ax[0,i].spines['left'].set_visible(False)
    ax[1,i].spines['right'].set_visible(False)
    ax[1,i].spines['left'].set_visible(False)
    
for i in range(0, len(ax[0])-1):
    d = .02 # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax[0,i].transAxes, color='k', clip_on=False)
    ax[0,i].plot((1-d,1+d), (-d,+d), **kwargs)
    ax[0,i].plot((1-d,1+d),(1-d,1+d), **kwargs)
    kwargs.update(transform=ax[0,i+1].transAxes)  # switch to the bottom axes
    ax[0,i+1].plot((-d,+d), (1-d,1+d), **kwargs)
    ax[0,i+1].plot((-d,+d), (-d,+d), **kwargs)
    
    kwargs = dict(transform=ax[1,i].transAxes, color='k', clip_on=False)
    ax[1,i].plot((1-d,1+d), (-d,+d), **kwargs)
    ax[1,i].plot((1-d,1+d),(1-d,1+d), **kwargs)
    kwargs.update(transform=ax[1,i+1].transAxes)  # switch to the bottom axes
    ax[1,i+1].plot((-d,+d), (1-d,1+d), **kwargs)
    ax[1,i+1].plot((-d,+d), (-d,+d), **kwargs)
# plt.tight_layout()
plt.show()

# %%
figure1 = plt.figure( figsize=(22, 12))
plt.plot(xt[0,:])
plt.plot(xt_mean[0,:], ':k')
