"""
This code belongs to the paper:
-- Probabilistic machine learning based predictive and interpretable digital
    twin for dynamical systems, authored by,
    Tapas Tripura, Aarya Sheetal Desai, Sondipon Adhikari, Souvik Chakraborty
   
*** This code is for the identification of Example 2: two-DOF dynamical system.
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import utils_alpha
import utils_data_alpha
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# %%

# The time parameters:
x0 = np.array([0.1, 0.1, 0, 0])
t0, tn, dt = 0, 1, 0.001

# Response generation:
xdt, xt, uv, t_eval = utils_data_alpha.twoD_duffing(t0, tn, x0, dt)

# %%
"""
# Gibbs sampling:
"""
# Parameter Initialisation:
MCMC = 3000 # No. of samples in Markov Chain,
burn_in = 1001
polyorder, tensor, harmonic, exponent = 6, 0, 0, 0
zstore_drift, Zmean_drift, theta_drift, mut_drift, sigt_drift = [], [], [], [], []
for i in  range(len(xdt)):
    print('Drift: state-',i)
    z1t, z2t, z3t, z4t, z5t, D = utils_alpha.sparse(xdt[i,:], xt, uv, polyorder, tensor, harmonic, exponent, MCMC, burn_in)
    zstore_drift.append(z1t)
    Zmean_drift.append(z2t)
    theta_drift.append(z3t)
    mut_drift.append(z4t)
    sigt_drift.append(z5t)
    
# %%
for i in range(len(xdt)):
    mut_drift[i][np.where(Zmean_drift[i] < 0.5)] = 0
    Zmean_drift[i][np.where(Zmean_drift[i] < 0.5)] = 0

for i in range(len(xdt)):
    mut_drift[i][0] = 0
    Zmean_drift[i][0] = 0

# %%
"""
Plotting commands:
"""
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24

figure1=plt.figure(figsize = (14, 10))
plt.subplot(211)
plt.stem(np.array(range(len(Zmean_drift[0]))), Zmean_drift[0], use_line_collection = True, linefmt='b', basefmt="k", markerfmt ='bo', label='Drift-1')
plt.axhline(y= 0.5, color='r', linestyle='-.')
plt.ylabel('PIP', fontweight='bold');
plt.title('(a)', fontweight='bold')
plt.grid(True); plt.ylim(0,1.05); plt.legend()
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(212)
plt.stem(np.array(range(len(Zmean_drift[1]))), Zmean_drift[1], use_line_collection = True, linefmt='b', basefmt="k", markerfmt ='bo', label='Drift-2')
plt.axhline(y= 0.5, color='r', linestyle='-.')
plt.ylabel('PIP', fontweight='bold');
plt.title('(b)', fontweight='bold')
plt.grid(True); plt.ylim(0,1.05); plt.legend()
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.tight_layout()
plt.show()

# %%
figure2=plt.figure(figsize = (16, 12))
plt.subplots_adjust(hspace=0.35, wspace=0.3)

plt.subplot(221)
xy = np.vstack([theta_drift[0][15,:], theta_drift[0][16,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(theta_drift[0][15,:], theta_drift[0][16,:], c=z, s=100)
plt.xlabel(' 'r'$\theta (x_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_2)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(a)', fontweight='bold')
plt.grid(True); #plt.xlim(-1010,-995)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(222)
xy = np.vstack([theta_drift[0][19,:], theta_drift[0][25,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(theta_drift[0][19,:], theta_drift[0][25,:], c=z, s=100)
plt.xlabel(' 'r'$\theta (x_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_1^3)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(b)', fontweight='bold')
plt.grid(True); #plt.xlim(-1010,-995); plt.ylim(-103000,-94000);
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(223)
xy = np.vstack([theta_drift[1][15,:], theta_drift[1][16,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(theta_drift[1][15,:], theta_drift[1][16,:], c=z, s=100)
plt.xlabel(' 'r'$\theta (x_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_2)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(c)', fontweight='bold')
plt.grid(True); #plt.xlim(-1010,-995)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(224)
xy = np.vstack([theta_drift[1][19,:], theta_drift[1][25,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(theta_drift[1][19,:], theta_drift[1][25,:], c=z, s=100)
plt.xlabel(' 'r'$\theta (x_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_1^3)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(d)', fontweight='bold')
plt.grid(True); #plt.xlim(-1010,-995); plt.ylim(-103000,-94000);
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.show()
