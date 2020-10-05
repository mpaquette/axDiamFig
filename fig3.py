#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import jnp_zeros

import matplotlib.pyplot as pl
import matplotlib.colors as colors
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

import scipy.stats as ss


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


D0 = 2.0e-9

# tweak the parameter range so that qmax give me something sensible
qmin = 0.01 # min quantile
qmax = 0.99 # max quantile


# analytical R_eff for gamma computed from moments
def reff_gamma(k, theta):
    return theta*((k+5)*(k+4)*(k+3)*(k+2))**0.25


def r_eff(counts, diams):
    # normalize counts
    normCounts = counts / counts.sum()
    # compute <r^6>
    r_mom6 = (normCounts*(diams/2.)**6).sum()
    # compute <r^2>
    r_mom2 = (normCounts*(diams/2.)**2).sum()
    # return r_eff = (<r^6>/<r^2>)^1/4
    return (r_mom6/r_mom2)**0.25


def get_gamma_binned_pdf(k, theta, maxq=0.99, N=100):
    rv = ss.gamma(k, loc=0, scale=theta) # define random variable for the diameters
    dmin = 0.1
    dmax = rv.ppf(maxq) # find d corresponding to max quartile

    ds_left = np.linspace(dmin, dmax, N, endpoint=True)
    spacing = ds_left[1] - ds_left[0] # get "bins" width
    ds_right = ds_left + spacing 
    ds_center = ds_left + (spacing/2)
    # Tai's method XD
    areas = (rv.pdf(ds_left) + rv.pdf(ds_right))*spacing/2.
    areas /= areas.sum() # normalize to sum 1 for our a purpose, so not technically a density anymore
    return ds_center, areas

def get_signal_fraction(radii, prob):
    # "volume" weigths
    crosssections = radii**2
    # signal fractions
    return (prob*crosssections)/crosssections


# GROUND TRUTH

# param
k = 2.25
theta = 0.4
peak = (k-1)*theta  

# count probabilities
ds_center, areas = get_gamma_binned_pdf(k, theta, maxq=qmax, N=100)
# signal fractions
signal_f = get_signal_fraction(ds_center/2., areas)
# generate signal for each diameter
signal = np.array([vangelderen_cylinder_perp_acq(D0, d*0.5e-6, acq, m_max=5) for d in ds_center]) # convert um diameter into m radius
# sum signals with the weigths
signal_full = (signal_f[:, None] * signal).sum(axis=0)

gt_deff = 2*reff_gamma(k, theta)
gt_deff_true = 2*r_eff(areas, ds_center)


# # plot grouth truth count distribution
# pl.figure()
# pl.plot(ds_center, areas)
# pl.title('GT, signal decay = [{:.1e}, {:.1e}, {:.1e}]'.format(*(1-signal_full)))
# pl.show()


krange = np.linspace(1.05, 9, 160, endpoint=True)
peakrange = np.linspace(0.05, 3, 119, endpoint=True)


# signal storage
data = np.zeros((krange.shape[0], peakrange.shape[0], acq.shape[0]))
dmin_data = np.zeros((krange.shape[0], peakrange.shape[0]))
dmax_data = np.zeros((krange.shape[0], peakrange.shape[0]))
k_data = np.zeros((krange.shape[0], peakrange.shape[0]))
t_data = np.zeros((krange.shape[0], peakrange.shape[0]))
p_data = np.zeros((krange.shape[0], peakrange.shape[0]))
data_deff_true = np.zeros((krange.shape[0], peakrange.shape[0]))

from time import time
startt = time()
for k_i, k_1 in enumerate(krange):
    print('{} / {}'.format(k_i, krange.shape[0]))
    for t_i, peak_1 in enumerate(peakrange):
        theta_1 = peak_1/(k_1-1)
        # count probabilities
        ds_center_1, areas_1 = get_gamma_binned_pdf(k_1, theta_1, maxq=qmax, N=100)
        data_deff_true[k_i, t_i] = 2*r_eff(areas_1, ds_center_1)
        # signal fractions
        signal_f_1 = get_signal_fraction(ds_center_1/2., areas_1)
        # generate signal for each diameter
        signal_1 = np.array([vangelderen_cylinder_perp_acq(D0, d*0.5e-6, acq, m_max=5) for d in ds_center_1]) # convert um diameter into m radius
        # sum signals with the weigths
        signal_full_1 = (signal_f_1[:, None] * signal_1).sum(axis=0)
        # logging
        data[k_i, t_i] = signal_full_1
        dmin_data[k_i, t_i] = ds_center_1.min()
        dmax_data[k_i, t_i] = ds_center_1.max()
        k_data[k_i, t_i] = k_1
        t_data[k_i, t_i] = theta_1
        p_data[k_i, t_i] = peak_1


endt = time()
print('time = {:.0f} seconds for {} gammas'.format(endt-startt, krange.shape[0]*peakrange.shape[0]))



# compute error
errors = np.sum(np.abs(data - signal_full), axis=2) / signal_full.shape[0]
data_deff = 2*reff_gamma(k_data, t_data)





qmaxplot = 0.999
Nplot = 1000

xmaxplot = 5
ymaxplot = 1.2


# TOP LEFT
k_t_l = 4.95
peak_t_l = 0.85
theta_t_l = peak_t_l / (k_t_l-1)

# BOTTOM LEFT
k_b_l = 1.25
peak_b_l = 0.15
theta_b_l = peak_b_l / (k_b_l-1)

# TOP RIGHT
k_t_r = 8.95
peak_t_r = 1.025
theta_t_r = peak_t_r / (k_t_r-1)

# BOTTOM RIGHT
k_b_r = k
peak_b_r = peak
theta_b_r = peak_b_r / (k_b_r-1)



import matplotlib.ticker as ticker
def fmt(x, pos):
    a, b = '{:.1f}'.format(100*x).split('.')
    return r'{:}.{:} \%'.format(a, b)


dpi = 100
pl.figure(figsize=(16, 9), dpi=dpi)
gs = gridspec.GridSpec(2, 3, wspace=0.35)
ax_main = pl.subplot(gs[0:2, 1])
levels = [0, 0.001, 0.005, 0.01, 0.05]

mycolormap = pl.cm.Blues
_colors = [mycolormap(i) for i in np.linspace(0, 1, len(levels))][::-1]
cp = pl.contourf(p_data, k_data, errors, levels, colors=_colors)
pl.xticks(fontsize=16)
pl.yticks(fontsize=16)

cbar = pl.colorbar(cp, format=ticker.FuncFormatter(fmt))
cbar.ax.tick_params(labelsize=16)

ax_main.set_aspect(aspect=1)
ax_main.set_xlabel(r'peak location ($\mu$m)', fontsize=20)
ax_main.set_ylabel(r'k (shape parameter)', fontsize=20)


cc_t_l = Circle((peak_t_l, k_t_l), radius=0.08, color='#00FF00') # green
cc_t_r = Circle((peak_t_r, k_t_r), radius=0.08, color='#FF00FF') # pink
cc_b_l = Circle((peak_b_l, k_b_l), radius=0.08, color='#FFFF00') # yellow
cc_b_r = Circle((peak_b_r, k_b_r), radius=0.08, color='#ff0000') # red
ax_main.add_patch(cc_t_l)
ax_main.add_patch(cc_t_r)
ax_main.add_patch(cc_b_l)
ax_main.add_patch(cc_b_r)

# extra thick axis spine are hidding bottom of plot
minus_y_buffer = 0.01

## top left
ax_t_l = pl.subplot(gs[0, 0])
ds_center_t_l, areas_t_l = get_gamma_binned_pdf(k_t_l, theta_t_l, maxq=qmaxplot, N=Nplot)
width_t_l = ds_center_t_l[2] - ds_center_t_l[1]
ax_t_l.bar(ds_center_t_l, areas_t_l/width_t_l, width_t_l, color=_colors[0], alpha=0.3)
ax_t_l.plot(ds_center_t_l, areas_t_l/width_t_l, color=_colors[0], linewidth=3)
ax_t_l.set_title(r'$\Gamma({:.2f}, {:.2f})$  (peak = ${:.2f}$ $\mu$m)'.format(k_t_l, theta_t_l, peak_t_l, 100*errors[list(np.round(krange, 3)).index(k_t_l), list(np.round(peakrange, 3)).index(peak_t_l)]), fontsize=16)
ax_t_l.set_xlim(0, xmaxplot)
ax_t_l.set_ylim(0-minus_y_buffer, ymaxplot)
pl.xticks(fontsize=13)
pl.yticks(fontsize=13)
for axis in ['top','bottom','left','right']:
    ax_t_l.spines[axis].set_linewidth(3)
    ax_t_l.spines[axis].set_color(cc_t_l.get_facecolor())


## bottom left
ax_b_l = pl.subplot(gs[1, 0])
ds_center_b_l, areas_b_l = get_gamma_binned_pdf(k_b_l, theta_b_l, maxq=qmaxplot, N=Nplot)
width_b_l = ds_center_b_l[2] - ds_center_b_l[1]
ax_b_l.bar(ds_center_b_l, areas_b_l/width_b_l, width_b_l, color=_colors[0], alpha=0.3)
ax_b_l.plot(ds_center_b_l, areas_b_l/width_b_l, color=_colors[0], linewidth=3)
ax_b_l.set_title(r'$\Gamma({:.2f}, {:.2f})$  (peak = ${:.2f}$ $\mu$m)'.format(k_b_l, theta_b_l, peak_b_l, 100*errors[list(np.round(krange, 3)).index(k_b_l), list(np.round(peakrange, 3)).index(peak_b_l)]), fontsize=16)
ax_b_l.set_xlim(0, xmaxplot)
ax_b_l.set_ylim(0-minus_y_buffer, ymaxplot)
pl.xticks(fontsize=13)
pl.yticks(fontsize=13)
for axis in ['top','bottom','left','right']:
    ax_b_l.spines[axis].set_linewidth(3)
    ax_b_l.spines[axis].set_color(cc_b_l.get_facecolor())

## top right
ax_t_r = pl.subplot(gs[0, 2])
ds_center_t_r, areas_t_r = get_gamma_binned_pdf(k_t_r, theta_t_r, maxq=qmaxplot, N=Nplot)
width_t_r = ds_center_t_r[2] - ds_center_t_r[1]
ax_t_r.bar(ds_center_t_r, areas_t_r/width_t_r, width_t_r, color=_colors[0], alpha=0.3)
ax_t_r.plot(ds_center_t_r, areas_t_r/width_t_r, color=_colors[0], linewidth=3)
ax_t_r.set_title(r'$\Gamma({:.2f}, {:.2f})$  (peak = ${:.2f}$ $\mu$m)'.format(k_t_r, theta_t_r, peak_t_r, 100*errors[list(np.round(krange, 3)).index(k_t_r), list(np.round(peakrange, 3)).index(peak_t_r)]), fontsize=16)
ax_t_r.set_xlim(0, xmaxplot)
ax_t_r.set_ylim(0-minus_y_buffer, ymaxplot)
pl.xticks(fontsize=13)
pl.yticks(fontsize=13)
for axis in ['top','bottom','left','right']:
    ax_t_r.spines[axis].set_linewidth(3)
    ax_t_r.spines[axis].set_color(cc_t_r.get_facecolor())


## bottom right
ax_b_r = pl.subplot(gs[1, 2])
ds_center_b_r, areas_b_r = get_gamma_binned_pdf(k_b_r, theta_b_r, maxq=qmaxplot, N=Nplot)
width_b_r = ds_center_b_r[2] - ds_center_b_r[1]
ax_b_r.bar(ds_center_b_r, areas_b_r/width_b_r, width_b_r, color=_colors[0], alpha=0.3)
ax_b_r.plot(ds_center_b_r, areas_b_r/width_b_r, color=_colors[0], linewidth=3)
ax_b_r.set_title(r'$\Gamma({:.2f}, {:.2f})$  (peak = ${:.2f}$ $\mu$m)'.format(k_b_r, theta_b_r, peak_b_r, 100*errors[list(np.round(krange, 3)).index(k_b_r), list(np.round(peakrange, 3)).index(peak_b_r)]), fontsize=16)
ax_b_r.set_xlim(0, xmaxplot)
ax_b_r.set_ylim(0-minus_y_buffer, ymaxplot)
pl.xticks(fontsize=13)
pl.yticks(fontsize=13)
for axis in ['top','bottom','left','right']:
    ax_b_r.spines[axis].set_linewidth(3)
    ax_b_r.spines[axis].set_color(cc_b_r.get_facecolor())


# pl.show()
pl.savefig("Figure_3.png")


