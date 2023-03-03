"""
This code belongs to the paper:
-- Probabilistic machine learning based predictive and interpretable digital
    twin for dynamical systems, authored by,
    Tapas Tripura, Aarya Sheetal Desai, Sondipon Adhikari, Souvik Chakraborty
   
*** This code is for the identification of Example 3: Degrading system.
    (Force is not measurable)
"""
    
from IPython import get_ipython
get_ipython().magic('reset -sf')

# %%

import numpy as np
import utils
import utils_data
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

# %%
"""
For the drift1: the crack propagation equation, identification :::
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# Response generation:
T = 1
dt= 0.001
x1, x2, x3 = 0.25, 0, 0 # initial displacement for duffing
xdt1, xdt2, _, y1, y2, y3, t_eval = utils_data.Crack_degradation(x1, x2, x3, T, dt)

ydata = [y1,y2]

MCMC = 5000 # No. of samples in Markov Chain,
burn_in = 1001
polyorder, tensor, harmonic, exponent = 3, 0, 0, 0

zstoredrift1, Zmeandrift1, thetadrift1, mutdrift1, sigtdrift1 = \
    utils.sparse(xdt1, ydata, polyorder, tensor, harmonic, exponent, MCMC, burn_in)

# Invoking PIP creteria with P(z_i < 0.5) = 0,
Zmeandrift1[np.where(Zmeandrift1 < 0.5)] = 0
mutdrift1[np.where(Zmeandrift1 < 0.5)] = 0

# %%
'defining extra exponential degradation term'
y3n = np.zeros(len(xdt1)+1)
y1_mean = np.mean(y1, axis = 0)
y2_mean = np.mean(y2, axis = 0)
for i in range(len(xdt1)):
    data = np.row_stack((y1_mean[i], y2_mean[i]))
    Dtemp, nl = utils.library(data, 3, 0, 0, 0)
    y3n[i+1] = y3n[i] + np.dot(Dtemp, mutdrift1)*dt
y3 = np.repeat(np.expand_dims(y3n,0),y1.shape[0],0)

#%%
"""
For the drift2 identification :::
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# Bayesian Interference:
ydata = [y1,y2,y3]
zstoredrift2, Zmeandrift2, thetadrift2, mutdrift2, sigtdrift2 = \
    utils.sparse(xdt2, ydata, polyorder, tensor, harmonic, 1, MCMC, burn_in)

# Invoking PIP creteria with P(z_i < 0.5) = 0,
Zmeandrift2[np.where(Zmeandrift2 < 0.5)] = 0
mutdrift2[np.where(Zmeandrift2 < 0.5)] = 0

# %%
"""
For the diffusion identification :::
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# Response generation:
x1, x2, x3 = 0.0001, 0, 0 # initial displacement condition
_, _, xbt, y1, y2, y3, t_eval = utils_data.Crack_degradation(x1, x2, x3, T, dt)

ydata = [y1,y2,y3]
zstorediff, Zmeandiff, thetadiff, mutdiff, sigtdiff = \
    utils.sparse(xbt, ydata, polyorder, tensor, harmonic, 1, MCMC, burn_in)

# Invoking PIP creteria with P(z_i < 0.5) = 0,
Zmeandiff[np.where(Zmeandiff < 0.5)] = 0
mutdiff[np.where(Zmeandiff < 0.5)] = 0

# Post processing:
mutdiff[np.where(np.diag(sigtdiff)>0.01)] = 0
Zmeandiff[np.where(np.diag(sigtdiff)>0.01)] = 0

# %%
"""
Plotting commands:
"""
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24

figure1=plt.figure(figsize = (14, 6))
plt.stem(np.array(range(len(Zmeandrift1))), Zmeandrift1, use_line_collection = True, linefmt='b', basefmt="k", markerfmt ='bo', label='Crack-path')
plt.stem(np.array(range(len(Zmeandrift2))), Zmeandrift2, use_line_collection = True, linefmt='g', basefmt="k", markerfmt ='gs', label='Drift')
plt.stem(np.array(range(len(Zmeandiff))), Zmeandiff, use_line_collection = True, linefmt='r', basefmt="k", markerfmt ='rd', label='Diffusion')

plt.axhline(y= 0.5, color='r', linestyle='-.')
plt.ylabel('PIP', fontweight='bold');
plt.title('(a)', fontweight='bold')
plt.grid(True); plt.ylim(0,1.05); plt.legend()
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.tight_layout()
plt.show()

# %%
figure2 = plt.figure(figsize = (16, 6))
plt.subplots_adjust(wspace=0.3)

plt.subplot(121)
xy = np.vstack([thetadrift1[3,:], thetadrift1[5,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift1[3,:], thetadrift1[5,:], c=z, s=100)
plt.xlabel(' 'r'$\theta (x_1^2)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_2^2)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(a) Crack-path', fontweight='bold')
plt.grid(True); plt.xlim(0.005,0.015); plt.ylim(0.009,0.011)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(122)
xy = np.vstack([thetadrift2[1,:], thetadrift2[29,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift2[1,:], thetadrift2[29,:], c=z, s=100)
plt.xlabel(' 'r'$\theta (x_1)_{drift}$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_1^3)_{drift}$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(b) Drift', fontweight='bold')
plt.grid(True); plt.xlim(-900,-1300); plt.ylim(-700,-1100);
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.show()

# %%
figure3 = plt.figure(figsize = (16, 6))
sns.distplot(thetadiff[0,:], kde_kws={"color": "b"},  hist_kws={"color": "r"})
plt.xlabel(' 'r'$\theta (1)$', fontweight='bold'); 
plt.title('(a) Diffusion', fontweight='bold'); plt.xlim(0.8,1.2);
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.grid(True);  
plt.show()
