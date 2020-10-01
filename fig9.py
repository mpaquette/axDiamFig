import numpy as np
from scipy.special import jnp_zeros

import matplotlib.pyplot as pl
import matplotlib.colors as colors
from matplotlib.patches import Circle

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}\usepackage{{amsfonts}}'

#  T^-1 s^-1
gamma = 42.515e6 * 2*np.pi

def vangelderen_cylinder_perp_ln_list(D, R, DELTA, delta, G, m_max=10):
    # returns the scaling factor and the list of each summation component for ln(M(DELTA,delta,G)/M(0))
    # D:= free diffusivity in m^2 s^-1
    # R:= cylinder radius in m
    # DELTA:= gradient separation in s
    # delta:= gradient width s
    # G:= gradient magnitude in T m^-1
    am_R = jnp_zeros(1,m_max)
    am = am_R / R
    am2 = am**2
    fac = -2*gamma**2*G**2/D**2
    comp = (1/(am**6*(am_R**2-1))) * (2*D*am2*delta - 2 + 2*np.exp(-D*am2*delta) + 2*np.exp(-D*am2*DELTA) - np.exp(-D*am2*(DELTA-delta)) - np.exp(-D*am2*(DELTA+delta)))
    return fac, comp



def vangelderen_cylinder_perp_ln(D, R, DELTA, delta, G, m_max=5):
    fac, comp = vangelderen_cylinder_perp_ln_list(D, R, DELTA, delta, G, m_max)
    return fac*np.sum(comp)


def vangelderen_cylinder_perp_acq(D, R, acq, m_max=5):
    S = []
    for acqpar in acq:
        G, delta, DELTA = acqpar
        lnS = vangelderen_cylinder_perp_ln(D, R, DELTA, delta, G, m_max)
        S.append(np.exp(lnS))
    return np.array(S)


# acquisitions parameters
# [G, delta, DELTA] in [T/m, s, s]
acq = np.array([[300e-3, 30e-3, 50e-3],
                [300e-3, 40e-3, 50e-3],
                [300e-3, 50e-3, 50e-3]])


# Test signal

# Fig2
# # BIG / IN-vivo
# D = 2.0e-9
# RR1 = 0.5*4.5e-6
# RR2 = 0.5*3.5e-6
# ff1 = 0.3

# Fig 9
# MEDIUM / IN-vivo
D = 2.0e-9
RR1 = 0.5*3.5e-6
RR2 = 0.5*2.5e-6
ff1 = 0.3

# Fig 10
# # SMALL / IN-vivo
# D = 2.0e-9
# RR1 = 0.5*2.5e-6
# RR2 = 0.5*1.5e-6
# ff1 = 0.3

# # MEDIUM / EX-vivo
# D = 0.66e-9
# RR1 = 0.5*2.5e-6
# RR2 = 0.5*1.5e-6
# ff1 = 0.3


# define diameter dictionary boundary and resolution
min_R = 0.025e-6/2.
max_R = 6.0e-6/2.
Rs = np.linspace(min_R, max_R, 1196, endpoint=True)

# generate Radius dictionary
signals = []
for R in Rs:
    S = vangelderen_cylinder_perp_acq(D, R, acq)
    signals.append(S)



def compute_2d_slice_dico_diff(S, dico, f1, errorfunc=lambda S,gt: np.sum(np.abs(S-gt))/3):
    err1 = []
    for dico_S1 in dico:
        err2 = []
        for dico_S2 in dico:
            dico_S = f1*dico_S1 + (1-f1)*dico_S2
            err2.append(errorfunc(S,dico_S))
        err1.append(err2)
    return np.array(err1)


# setting up fractions for 2 cylinders experiment
min_f = 0.1
max_f = 0.5
x1 = 3
x2 = 3
fs = np.linspace(min_f, max_f, x1*x2, endpoint=True)

# ground truth
S1 = vangelderen_cylinder_perp_acq(D, RR1, acq)
S2 = vangelderen_cylinder_perp_acq(D, RR2, acq)
S = ff1*S1 + (1-ff1)*S2


# compute errors for the whole dictionary
err_S = []
for f in fs:
    err_S_f = compute_2d_slice_dico_diff(S, signals, f)
    err_S.append(err_S_f)


err_S_array = np.array(err_S)
tt = 0.03
ttt = np.min(err_S_array) + tt
minV = err_S_array.min()
maxV = err_S_array.max()



# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def plot_err(err, minV, maxV, divV):
    elev_min = minV
    elev_max = maxV
    mid_val = divV
    cmap=pl.cm.RdBu_r # set the colormap to something diverging
    fig = pl.figure()
    pl.imshow(err, cmap=cmap, clim=(elev_min, elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max),extent=[(Rs*1e6).min(),(Rs*1e6).max(),(Rs*1e6).max(),(Rs*1e6).min()])
    pl.colorbar()
    return fig



import matplotlib.ticker as ticker
def fmt(x, pos):
    a, b = '{:.1f}'.format(100*x).split('.')
    return r'{:}.{:} \%'.format(a, b)


levels = [0, 0.001, 0.005, 0.01, 0.05]

# nicely distinguisable from set1 categorical
_colors = ['#984ea3', '#4daf4a', '#377eb8', '#ff7f00', '#e41a1c']


dpi = 600
fig, axes = pl.subplots(nrows=3, ncols=3, sharex=False, sharey=False, figsize=(14,12), dpi=dpi)
fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.5)

for ix,iy in np.ndindex((x1,x2)):
    i = ix*x2 + iy
    f = fs[i]
    axs = axes[ix, iy]
    cp = axs.contourf(Rs*2e6, Rs*2e6, err_S[i], levels, colors=_colors)

    if np.abs(f-ff1)<0.01:
        # dot with thick outline achieved by stacking 2 circle of different radius
        cc_gt_outer = Circle((RR2*2e6, RR1*2e6), radius=0.14, color='black')
        axs.add_patch(cc_gt_outer)
        cc_gt_inner = Circle((RR2*2e6, RR1*2e6), radius=0.06, color='white')
        axs.add_patch(cc_gt_inner)

    axs.set_aspect(aspect=1)

    axs.set_xlabel(r'$d_2$ ($\mu$m)', fontsize=16)
    axs.set_ylabel(r'$d_1$ ($\mu$m)', fontsize=16)
    axs.set_xticks(range(1,7))
    axs.set_yticks(range(1,7))
    axs.set_xticklabels(range(1,7), fontsize=14)
    axs.set_yticklabels(range(1,7), fontsize=14)

    axs.set_title(r'{:.0f}\% $d_1$ + {:.0f}\% $d_2$'.format(f*100, (1-f)*100), fontsize=16)

cbar = fig.colorbar(cp, ax=axes.ravel().tolist(), format=ticker.FuncFormatter(fmt))
cbar.ax.tick_params(labelsize=18)


# pl.show()
pl.savefig("Figure_9.png") # MEDIUM / IN-vivo

