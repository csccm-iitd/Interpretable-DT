"""
This code belongs to the paper:
-- Probabilistic machine learning based predictive and interpretable digital
    twin for dynamical systems, authored by,
    Tapas Tripura, Aarya Sheetal Desai, Sondipon Adhikari, Souvik Chakraborty
   
*** This code is for the identification of Example 1: SDOF dynamical system.
    (Force is not measurable)
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

# %%

import numpy as np
import utils
import utils_data
import matplotlib.pyplot as plt
import seaborn as sns

# %%

"""
For the drift identification :::
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# Response generation:
T = 1
dt= 0.001
x1, x2 = 0.1, 0 # initial displacement for duffing
xdt, _, y1, y2, t_eval = utils_data.duffing(x1, x2, T, dt)
ydata = [y1,y2]

MCMC = 5000 # No. of samples in Markov Chain,
burn_in = 1001
polyorder, tensor, harmonic, exponent = 6, 1, 0, 0

zstoredrift, Zmeandrift, thetadrift, mutdrift, sigtdrift = \
    utils.sparse(xdt, ydata, polyorder, tensor, harmonic, exponent, MCMC, burn_in)

# Invoking PIP creteria with P(z_i < 0.5) = 0,
Zmeandrift[np.where(Zmeandrift < 0.5)] = 0
mutdrift[np.where(Zmeandrift < 0.5)] = 0

# %%
"""
For the diffusion identification :::
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# Response generation:
x1, x2 = 0, 0 # initial displacement condition
_, bxt, y1, y2, t_eval = utils_data.duffing(x1, x2, T, dt)
ydata=[y1,y2]

zstorediff, Zmeandiff, thetadiff, mutdiff, sigtdiff = \
    utils.sparse(bxt, ydata, polyorder, tensor, harmonic, exponent, MCMC, burn_in)

# Invoking PIP creteria with P(z_i < 0.5) = 0,
Zmeandiff[np.where(Zmeandiff < 0.5)] = 0
mutdiff[np.where(Zmeandiff < 0.5)] = 0

# Post processing:
mutdiff[np.where(np.diag(sigtdiff)>0.001)] = 0
Zmeandiff[np.where(np.diag(sigtdiff)>0.001)] = 0
munormal = mutdiff

# %%
"""
Plotting commands:
"""
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24
plt.rcParams['font.weight'] = 'bold'

figure1=plt.figure(figsize = (14, 6))
nl = Zmeandrift.shape[0]
xr = np.array(range(nl))
plt.stem(xr, Zmeandrift, use_line_collection = True, linefmt='b', basefmt="k", markerfmt ='bo', label='Drift')
plt.stem(xr, Zmeandiff, use_line_collection = True, linefmt='r', basefmt="k", markerfmt ='rs', label='Diffusion')
plt.axhline(y= 0.5, color='r', linestyle='-.')
plt.ylabel('PIP', fontweight='bold');
plt.title('(a)', fontweight='bold')
plt.grid(True); plt.legend()
plt.ylim(0,1.05)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.tight_layout()
plt.show()

# %%
figure2 = plt.figure(figsize = (16, 6))
plt.subplot(121)
ax=sns.distplot(thetadrift[6,:], kde_kws={"color": "r"},  hist_kws={"color": "orange"})
plt.xlabel(' 'r'$\theta (x_1^3)$', fontweight='bold'); 
plt.title('(a) Drift', fontweight='bold'); plt.xlim(-1.01*1e5,-0.99*1e5);
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.grid(True);

plt.subplot(122)
ax=sns.distplot(thetadiff[0,:], kde_kws={"color": "b"},  hist_kws={"color": "b"})
plt.xlabel(' 'r'$\theta (1)$', fontweight='bold'); 
plt.title('(b) Diffusion', fontweight='bold'); plt.xlim(0.2,0.3);
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.grid(True);

plt.tight_layout()
plt.show()
