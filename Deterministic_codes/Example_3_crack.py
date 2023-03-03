"""
This code belongs to the paper:
-- Probabilistic machine learning based predictive and interpretable digital
    twin for dynamical systems, authored by,
    Tapas Tripura, Aarya Sheetal Desai, Sondipon Adhikari, Souvik Chakraborty
   
*** This code is for the identification of Example 3: Degrading system.
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
x0 = np.array([1, 0, 0])
t0, tn, dt = 0, 1, 0.001

# Response generation:
xdt1, xdt2, xt, uv, t_eval = utils_data_alpha.crack_degradation(t0, tn, x0, dt)

# %%
"""
# Identification of the degradation due to fatigue crack:
"""
# Parameter Initialisation:
MCMC = 3000 # No. of samples in Markov Chain,
burn_in = 1001
polyorder, tensor, harmonic, exponent = 3, 0, 0, 0
print('Drift: Crack: ')
zstore_drift1, Zmean_drift1, theta_drift1, mut_drift1, sigt_drift1 = \
    utils_alpha.sparse(xdt1, xt[:2, :], np.column_stack((uv)), polyorder, tensor, harmonic, exponent, MCMC, burn_in)

mut_drift1[np.where(Zmean_drift1 < 0.5)] = 0
Zmean_drift1[np.where(Zmean_drift1 < 0.5)] = 0

# %%
'defining extra exponential degradation term'
q = np.zeros(len(xdt1))
for i in range(len(xdt1)-1):
    data = np.row_stack((xt[0, i], xt[1, i]))
    Dtemp, nl = utils_alpha.library(data, uv[i], polyorder, tensor, harmonic, 0)
    q[i+1] = q[i] + np.dot(Dtemp, mut_drift1)*dt

figure=plt.figure(figsize = (14, 6))
plt.plot(q, label='Identified'); plt.plot(xt[2,:],':r', label='True');
plt.legend(); plt.grid(True)
plt.ylabel('z(t)'); plt.xlabel('Time (s)'); plt.margins(0)
plt.title('(a)', fontweight='bold')
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.tight_layout()
plt.show()

# %%
print('Drift: state: ')
zstore_drift2, Zmean_drift2, theta_drift2, mut_drift2, sigt_drift2 = \
    utils_alpha.sparse(xdt2, xt, np.column_stack((uv)), polyorder, tensor, harmonic, 1, MCMC, burn_in)

mut_drift2[np.where(Zmean_drift2 < 0.5)] = 0
Zmean_drift2[np.where(Zmean_drift2 < 0.5)] = 0

# %%
"""
Plotting commands:
"""
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24

figure1=plt.figure(figsize = (14, 6))
plt.stem(np.array(range(len(Zmean_drift1))), Zmean_drift1, use_line_collection = True, linefmt='b', basefmt="k", markerfmt ='bo', label='Crack-path')
plt.stem(0.15+np.array(range(len(Zmean_drift2))), Zmean_drift2, use_line_collection = True, linefmt='g', basefmt="k", markerfmt ='gs', label='Drift')

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
xy = np.vstack([theta_drift1[3,:], theta_drift1[5,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(theta_drift1[3,:], theta_drift1[5,:], c=z, s=100)
plt.xlabel(' 'r'$\theta (x_1^2)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_2^2)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(a) Crack-path', fontweight='bold')
plt.grid(True); #plt.xlim(0.005,0.015); plt.ylim(0.009,0.011)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(122)
xy = np.vstack([theta_drift2[1,:], theta_drift2[23,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(theta_drift2[1,:], theta_drift2[23,:], c=z, s=100)
plt.xlabel(' 'r'$\theta (x_1)_{drift}$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_1^3)_{drift}$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(b) Drift', fontweight='bold')
plt.grid(True); #plt.xlim(-900,-1300); plt.ylim(-700,-1100);
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.show()
