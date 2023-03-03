"""
This code belongs to the paper:
-- Probabilistic machine learning based predictive and interpretable digital
    twin for dynamical systems, authored by,
    Tapas Tripura, Aarya Sheetal Desai, Sondipon Adhikari, Souvik Chakraborty
   
*** This code is for the identification of Example 1: SDOF dynamical system.
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import utils_alpha
import utils_data_alpha
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

np.random.seed(0)
# %%

# The time parameters:
x0 = np.array([0.1, 0])
t0, tn, dt = 0, 1, 0.001

# Response generation:
xdt, xt, uv, t_eval = utils_data_alpha.duffing(t0, tn, x0, dt)

# %%
"""
# Gibbs sampling:
"""
# Parameter Initialisation:
MCMC = 3000 # No. of samples in Markov Chain,
burn_in = 1001
polyorder, tensor, harmonic, exponent = 6, 0, 0, 0
zstore_drift, Zmean_drift, theta_drift, mut_drift, sigt_drift = [], [], [], [], []
for i in  range(1):
    print('Drift: state-',i)
    z1t, z2t, z3t, z4t, z5t = \
        utils_alpha.sparse(xdt, xt, np.column_stack((uv)), polyorder, tensor, harmonic, exponent, MCMC, burn_in)
    zstore_drift.append(z1t)
    Zmean_drift.append(z2t)
    theta_drift.append(z3t)
    mut_drift.append(z4t)
    sigt_drift.append(z5t)
    
# %%
for i in range(1):
    mut_drift[i][np.where(Zmean_drift[i] < 0.5)] = 0
    Zmean_drift[i][np.where(Zmean_drift[i] < 0.5)] = 0

# %%
"""
Plotting commands:
"""
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24
plt.rcParams['font.weight'] = 'bold'

figure1 = plt.subplots(figsize = (14, 6))
yr = 0.5*np.ones(len(Zmean_drift[0]))
xr = np.array(range(0, len(Zmean_drift[i])))
plt.stem(xr, Zmean_drift[i], use_line_collection = True, linefmt='b', basefmt="k", markerfmt ='bo')
plt.plot(xr, yr, 'r')
plt.legend(["Identified states", "P(Y=0.5)"])
plt.xlabel('Library functions'); 
plt.ylabel('PIP'); plt.grid(True);
plt.ylim([0,1.05])
# plt.tight_layout()
plt.show()

# %%
figure2, axs2 = plt.subplots(figsize = (10, 8))

xy = np.vstack([theta_drift[0][6,:], theta_drift[0][30,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(theta_drift[0][6,:], theta_drift[0][30,:], c=z, s=100)
plt.xlabel(' 'r'$\theta (x_1^3)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (f_t)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(a)', fontweight='bold')
plt.grid(True); 

plt.tight_layout()
plt.show()
