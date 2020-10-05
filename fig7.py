#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import jnp_zeros

import pylab as pl
import matplotlib.ticker as ticker

from vangelderen import vangelderen_cylinder_perp
from scheme import expand_scheme, remove_unphysical

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}\usepackage{{amsfonts}}\usepackage{{siunitx}}'


D_invivo = 2e-9

# scanning param
list_G = [0.3]
list_DELTA = np.linspace(10e-3, 50e-3, 9)
spacing_DELTA = list_DELTA[1] - list_DELTA[0]
list_delta = np.linspace(10e-3, 50e-3, 9)
spacing_delta = list_delta[1] - list_delta[0]
scheme = expand_scheme(list_G, list_DELTA, list_delta)

# radius to simulate 
Rs = np.array([0.25, 0.5, 1, 2])*1e-6
n = np.sqrt(len(Rs))
ny = int(np.ceil(n))
nx = int(np.floor(n))
if nx*ny < len(Rs):
    nx += 1



# print all digit before decimal and up to the first 2 decimal
def fmt(x, pos):
    a, b = '{:.15f}'.format(x).split('.')
    # pre decimal
    a = int(a)
    # search for first non zero decimal
    notZero = np.array([digt!='0' for digt in b])
    posFirstDigit = np.where(notZero)[0][0]
    # check for rounding with 3rd digit
    if int(b[posFirstDigit+2]) < 5:
        c = '0'*posFirstDigit + b[posFirstDigit] + b[posFirstDigit+1]
    else:
        # we SHOULD check if b[posFirstDigit+1]+1 is 10, and if it is we increase b[posFirstDigit] by one and if it is also 10 now ....
        # but I wont
        c = '0'*posFirstDigit + b[posFirstDigit] + str(int(b[posFirstDigit+1])+1)

    return r'{:}.{:} \%'.format(a, c)



textfs = 20
dpi = 100
pl.figure(figsize=(14,12), dpi=dpi)
pl.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.35)
for iR,R in enumerate(Rs):
    S = vangelderen_cylinder_perp(D_invivo, R, scheme, 50)

    pl.subplot(nx, ny, iR+1)

    tmp = S.reshape(len(list_DELTA), len(list_delta))
    mask = np.isinf(tmp)
    # percent
    tmp2 = 100*np.ma.array(1-tmp, mask=mask)

    pl.imshow(tmp2, interpolation='nearest', extent=[1e3*(list_delta.min()-spacing_delta/2.),1e3*(list_delta.max()+spacing_delta/2.),1e3*(list_DELTA.max()+spacing_DELTA/2.),1e3*(list_DELTA.min()-spacing_DELTA/2.)])
    cbar = pl.colorbar(format=ticker.FuncFormatter(fmt))

    cbar.set_ticks([tmp2.min(), tmp2.max()])

    cbar.ax.tick_params(labelsize=textfs)
    pl.title(r'd = {:.1f} $\mu$m'.format(R*2e6), fontsize=textfs+2)
    pl.xlabel(r'$\delta$ (ms)', fontsize=textfs)
    pl.ylabel(r'$\Delta$ (ms)', fontsize=textfs)
    pl.xticks(fontsize=textfs-4)
    pl.yticks(fontsize=textfs-4)

# pl.show()
pl.savefig("Figure_7.png")




