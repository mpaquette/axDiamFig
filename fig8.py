import numpy as np

from scipy.stats import gamma 

import pylab as pl
from matplotlib import rc
rc('text', usetex=True)





def D_perp_intra_world(DELTA, delta, f_ex, D_inf_ex, r_app, D_0):
    # DELTA: big delta (ms)
    # delta: small delta (ms)
    # f_ex: extra axonal space volume fraction
    # f_in: intra axonal space volume fraction
    f_in = 1-f_ex
    # D_inf_ex: long-time limit extra axonal space diffusivity (um^2/ms)
    # r_app: apparent axon radius (um)
    # for distributions: r_app = <r^6>/<r^2>
    # D_0: intra axonal unrestricted diffusivity (um^2/ms)
    c = (7/48.)*(f_in*r_app**4)/D_0
    tau = DELTA - delta/3.
    return f_ex*D_inf_ex + (c/(delta*tau))

def D_perp_extra_world(DELTA, delta, f_ex, D_inf_ex, A):
    # DELTA: big delta (ms)
    # delta: small delta (ms)
    # f_ex: extra axonal space volume fraction
    # D_inf_ex: long-time limit extra axonal space diffusivity (um^2/ms)
    # A: (for now mysterious) disorder parameter
    c_prime = f_ex*A
    tau = DELTA - delta/3.
    return f_ex*D_inf_ex + c_prime*((np.log(DELTA/delta)+1.5)/tau)


def r_app_from_extra(DELTA, delta, f_ex, D_inf_ex, D_0, A):
    f_in = 1-f_ex
    c_prime = f_ex*A
    c = c_prime*(np.log(DELTA/delta)+1.5)*delta
    r_app = ((48/7.)*(D_0/f_in)*c)**0.25
    return r_app


# param
D_0 = 2.
D_inf_ex = 0.5


N_DELTAs = 96
DELTAs = np.linspace(5,100,N_DELTAs)
N_deltas = 46
deltas = np.linspace(5,50,N_deltas)



unphysical_mask = np.zeros((N_DELTAs, N_deltas), dtype=np.bool)


# D_perp_extra_world
As = np.array([0.25, 0.5, 1., 2.])
N_As = As.shape[0]
f_exs = np.array([0.25, 0.5, 0.75])
N_f_exs = f_exs.shape[0]


extra_D = np.zeros((N_As, N_f_exs, N_DELTAs, N_deltas))
fake_r_app_extra = np.zeros((N_As, N_f_exs, N_DELTAs, N_deltas))

extra_D_vmin = np.inf*np.ones((N_As, N_f_exs))
fake_r_app_extra_vmin = np.inf*np.ones((N_As, N_f_exs))

fake_r_app_extra_vmax = -np.inf*np.ones((N_As, N_f_exs))


for i_As in range(N_As):
    A = As[i_As]
    for i_f_exs in range(N_f_exs):
        f_ex = f_exs[i_f_exs]
        for i_DELTAs in range(N_DELTAs):
            DELTA = DELTAs[i_DELTAs]
            for i_deltas in range(N_deltas):
                delta = deltas[i_deltas]
                if DELTA >= delta:
                    tmp = D_perp_extra_world(DELTA, delta, f_ex, D_inf_ex, A)
                    extra_D[i_As, i_f_exs, i_DELTAs, i_deltas] = tmp
                    if tmp < extra_D_vmin[i_As, i_f_exs]:
                        extra_D_vmin[i_As, i_f_exs] = tmp

                    tmp = r_app_from_extra(DELTA, delta, f_ex, D_inf_ex, D_0, A)
                    fake_r_app_extra[i_As, i_f_exs, i_DELTAs, i_deltas] = tmp
                    if tmp < fake_r_app_extra_vmin[i_As, i_f_exs]:
                        fake_r_app_extra_vmin[i_As, i_f_exs] = tmp
                    if tmp > fake_r_app_extra_vmax[i_As, i_f_exs]:
                        fake_r_app_extra_vmax[i_As, i_f_exs] = tmp

                else:
                    unphysical_mask[i_DELTAs, i_deltas] = True


import matplotlib.ticker as ticker
def fmt(x, pos):
    return r'{:.0f} $\mu$m'.format(x)

dpi = 600
fig, axes = pl.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(16, 7), dpi=dpi)
for i_f_exs in range(N_f_exs):
    f_ex = f_exs[i_f_exs]
    for i_As in range(N_As):
        A = As[i_As]
        axs = axes[i_f_exs, i_As]

        plot_data = np.ma.array(2*fake_r_app_extra[i_As, i_f_exs], mask=unphysical_mask)

        im = axs.imshow(plot_data.T, interpolation='none', extent=[DELTAs.min(),DELTAs.max(),deltas.max(),deltas.min()], vmin=fake_r_app_extra_vmin.min(), vmax=fake_r_app_extra_vmax.max())

        axs.set_xticks([5, 25, 50, 75, 100])
        axs.set_yticks([5, 25, 50])
        axs.set_xticklabels([5, 25, 50, 75, 100], fontsize=16)
        axs.set_yticklabels([5, 25, 50], fontsize=16)

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), format=ticker.FuncFormatter(fmt))
cbar.ax.tick_params(labelsize=16)

for i_f_exs in range(N_f_exs):
    f_ex = f_exs[i_f_exs]
    fig.text(0.06, (i_f_exs+1)/float(N_f_exs+1), r'$f_{{ex}} = {}$'.format(f_ex), va='center', fontsize=20, rotation='vertical')

for i_As in range(N_As):
    A = As[i_As]
    fig.text((i_As+1.1)/float(N_As+2), 0.88, r'$A = {}$'.format(A), ha='center', fontsize=20)

axes[N_f_exs-1, 0].set_xlabel(r'$\Delta$ (ms)', fontsize=18)
axes[N_f_exs-1, 0].set_ylabel(r'$\delta$ (ms)', fontsize=18)

# pl.show()
pl.savefig("Figure_8.png")


