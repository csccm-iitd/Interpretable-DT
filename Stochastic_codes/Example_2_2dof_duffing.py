"""
This code belongs to the paper:
-- Probabilistic machine learning based predictive and interpretable digital
    twin for dynamical systems, authored by,
    Tapas Tripura, Aarya Sheetal Desai, Sondipon Adhikari, Souvik Chakraborty
   
*** This code is for the identification of Example 2: two-DOF dynamical system.
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
For the drift1 identification :::
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# Response generation:
T = 1
dt= 0.001
x1, x2, x3, x4 = 0.1, 0.1, 0, 0 # initial displacement for duffing
xdt1, _, xdt2, _, y1, y2, y3, y4, t_eval = utils_data.twoDOF_duffing(x1, x2, x3, x4, T, dt)
ydata = [y1, y2, y3, y4]

MCMC = 5000 # No. of samples in Markov Chain,
burn_in = 1001
polyorder, tensor, harmonic, exponent = 4, 0, 0, 0

zstoredrift1, Zmeandrift1, thetadrift1, mutdrift1, sigtdrift1 = \
    utils.sparse(xdt1, ydata, polyorder, tensor, harmonic, exponent, MCMC, burn_in)

# Invoking PIP creteria with P(z_i < 0.5) = 0,
Zmeandrift1[np.where(Zmeandrift1 < 0.5)] = 0
mutdrift1[np.where(Zmeandrift1 < 0.5)] = 0

#%%
"""
For the drift2 identification :::
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
zstoredrift2, Zmeandrift2, thetadrift2, mutdrift2, sigtdrift2 = \
    utils.sparse(xdt2, ydata, polyorder, tensor, harmonic, exponent, MCMC, burn_in)

# Invoking PIP creteria with P(z_i < 0.5) = 0,
Zmeandrift2[np.where(Zmeandrift2 < 0.5)] = 0
mutdrift2[np.where(Zmeandrift2 < 0.5)] = 0

# %%
"""
For the diffusion1 identification :::
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# Response generation:
x1, x2, x3, x4 = 0.001, 0.001, 0, 0 # initial displacement condition
_, xt1, _, xt2, y1, y2, y3, y4, t_eval = utils_data.twoDOF_duffing(x1, x2, x3, x4, T, dt)

ydata = [y1,y2,y3,y4]
polyorder, tensor, harmonic, exponent = 3, 0, 0, 0
zstorediff1, Zmeandiff1, thetadiff1, mutdiff1, sigtdiff1 = \
    utils.sparse(xt1, ydata, polyorder, tensor, harmonic, exponent, MCMC, burn_in)

# Invoking PIP creteria with P(z_i < 0.5) = 0,
Zmeandiff1[np.where(Zmeandiff1 < 0.5)] = 0
mutdiff1[np.where(Zmeandiff1 < 0.5)] = 0

mutdiff1[np.where(np.diag(sigtdiff1)>0.01)] = 0
Zmeandiff1[np.where(np.diag(sigtdiff1)>0.01)] = 0

# %%
"""
For the diffusion2 identification :::
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# Bayesian Interference:
zstorediff2, Zmeandiff2, thetadiff2, mutdiff2, sigtdiff2 = \
    utils.sparse(xt2, ydata, polyorder, tensor, harmonic, exponent, MCMC, burn_in)

# Invoking PIP creteria with P(z_i < 0.5) = 0,
Zmeandiff2[np.where(Zmeandiff2 < 0.5)] = 0
mutdiff2[np.where(Zmeandiff2 < 0.5)] = 0

mutdiff2[np.where(np.diag(sigtdiff2)>0.01)] = 0
Zmeandiff2[np.where(np.diag(sigtdiff2)>0.01)] = 0

#%%
"""
Plotting commands:
"""

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24

figure1=plt.figure(figsize = (14, 10))
plt.subplot(211)
plt.stem(np.array(range(len(Zmeandrift1))), Zmeandrift1, use_line_collection = True, linefmt='b', basefmt="k", markerfmt ='bo', label='Drift-1')
plt.stem(np.array(range(len(Zmeandiff1))), Zmeandiff1, use_line_collection = True, linefmt='r', basefmt="k", markerfmt ='rs', label='Diffusion-1')

plt.axhline(y= 0.5, color='r', linestyle='-.')
plt.ylabel('PIP', fontweight='bold');
plt.title('(a)', fontweight='bold')
plt.grid(True); plt.ylim(0,1.05); plt.legend()
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(212)
plt.stem(np.array(range(len(Zmeandrift2))), Zmeandrift2, use_line_collection = True, linefmt='b', basefmt="k", markerfmt ='bo', label='Drift-2')
plt.stem(np.array(range(len(Zmeandiff2))), Zmeandiff2, use_line_collection = True, linefmt='r', basefmt="k", markerfmt ='rs', label='Diffusion-2')

plt.axhline(y= 0.5, color='r', linestyle='-.')
plt.ylabel('PIP', fontweight='bold');
plt.title('(b)', fontweight='bold')
plt.grid(True); plt.ylim(0,1.05); plt.legend()
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.tight_layout()
plt.show()

# %%
figure2 = plt.figure(figsize = (16, 6))
plt.subplot(121)
ax=sns.distplot(thetadiff1[0,:], kde_kws={"color": "r"},  hist_kws={"color": "orange"})
plt.xlabel(' 'r'$\theta (1)$', fontweight='bold'); 
plt.title('(a) Diffusion-1', fontweight='bold'); plt.xlim(0.25,0.29);
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.grid(True);

plt.subplot(122)
ax=sns.distplot(thetadiff2[0,:], kde_kws={"color": "b"},  hist_kws={"color": "b"})
plt.xlabel(' 'r'$\theta (1)$', fontweight='bold'); 
plt.title('(b) Diffusion-2', fontweight='bold'); plt.xlim(0.23,0.27);
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.grid(True);

plt.tight_layout()
plt.show()

# %%
figure3=plt.figure(figsize = (16, 12))
plt.subplots_adjust(hspace=0.35, wspace=0.3)

plt.subplot(221)
xy = np.vstack([thetadrift1[15,:], thetadrift1[16,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift1[15,:], thetadrift1[16,:], c=z, s=100)
plt.xlabel(' 'r'$\theta (x_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_2)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(a)', fontweight='bold')
plt.grid(True); #plt.xlim(-1010,-995)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(222)
xy = np.vstack([thetadrift1[19,:], thetadrift1[25,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift1[19,:], thetadrift1[25,:], c=z, s=100)
plt.xlabel(' 'r'$\theta (x_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_1^3)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(b)', fontweight='bold')
plt.grid(True); #plt.xlim(-1010,-995); plt.ylim(-103000,-94000);
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(223)
xy = np.vstack([thetadrift2[15,:], thetadrift2[16,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift2[15,:], thetadrift2[16,:], c=z, s=100)
plt.xlabel(' 'r'$\theta (x_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_2)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(c)', fontweight='bold')
plt.grid(True); #plt.xlim(-1010,-995)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(224)
xy = np.vstack([thetadrift2[19,:], thetadrift2[25,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift2[19,:], thetadrift2[25,:], c=z, s=100)
plt.xlabel(' 'r'$\theta (x_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_1^3)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(d)', fontweight='bold')
plt.grid(True); #plt.xlim(-1010,-995); plt.ylim(-103000,-94000);
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.show()
