#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl

from matplotlib import rc
rc('text', usetex=True)

# m
diffusivity = 2.0e-9

# s
diff_time = 80.0e-3
t_step = 10.0e-6
N_init = 10000
N_particule = N_init



## TODO have this actually somewhere
displacement_1 = np.load('displacement_D2p0_R5p0.npy')
times_1 = np.load('times_D2p0_R5p0.npy')
cyl_radius_1 = 5.0e-6
msd_1 = (1e12)*(displacement_1**2).mean(axis=1)
limit_1 = 0.5*((1e6)*cyl_radius_1)**2

displacement_2 = np.load('displacement_D2p0_R2p5.npy')
times_2 = np.load('times_D2p0_R2p5.npy')
cyl_radius_2 = 2.5e-6
msd_2 = (1e12)*(displacement_2**2).mean(axis=1)
limit_2 = 0.5*((1e6)*cyl_radius_2)**2

displacement_3 = np.load('displacement_D2p0_R1p0.npy')
times_3 = np.load('times_D2p0_R1p0.npy')
cyl_radius_3 = 1.0e-6
msd_3 = (1e12)*(displacement_3**2).mean(axis=1)
limit_3 = 0.5*((1e6)*cyl_radius_3)**2

displacement_4 = np.load('displacement_D2p0_R0p5.npy')
times_4 = np.load('times_D2p0_R0p5.npy')
cyl_radius_4 = 0.5e-6
msd_4 = (1e12)*(displacement_4**2).mean(axis=1)
limit_4 = 0.5*((1e6)*cyl_radius_4)**2

unrestricted = (1e12)*2*diffusivity*times_1[1:]



# curves color
c1 = (1,0,0)
c2 = (0,1,0)
c3 = (0,0,1)
c4 = (1,0,1)
c7 = (0,0,0)


tl = times_1.shape[0]



label_size = 24
pl.rcParams['xtick.labelsize'] = label_size
pl.rcParams['ytick.labelsize'] = label_size

dpi = 100
fig, ax1 = pl.subplots(figsize=(16,9), dpi=dpi)
ax1.plot((1e3)*times_1[1:tl], unrestricted[:tl-1], c=c7, label='Free')
ax1.plot((1e3)*times_1[1:tl], msd_1[:tl-1], c=c1, label=r'd = {:.0f} $\mu$m'.format((1e6)*cyl_radius_1*2))
ax1.axhline(limit_1, c=c1)
ax1.plot((1e3)*times_1[1:tl], msd_2[:tl-1], c=c2, label=r'd = {:.0f} $\mu$m'.format((1e6)*cyl_radius_2*2))
ax1.axhline(limit_2, c=c2)
ax1.plot((1e3)*times_1[1:tl], msd_3[:tl-1], c=c3, label=r'd = {:.0f} $\mu$m'.format((1e6)*cyl_radius_3*2))
ax1.axhline(limit_3, c=c3)
ax1.plot((1e3)*times_1[1:tl], msd_4[:tl-1], c=c4, label=r'd = {:.0f} $\mu$m'.format((1e6)*cyl_radius_4*2))
ax1.axhline(limit_4, c=c4)

ax1.set_xlabel('Time (ms)', size=24)
ax1.set_ylabel(r'Mean Squared Displacement ($\mu$m$^2$)', size=24)
ax1.legend(fontsize=24)
ax1.set_ylim(bottom=0, top=msd_1[:tl-1].max()*1.2)

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.3, 0.35, 0.28, 0.28]
ax2 = fig.add_axes([left, bottom, width, height])

# shortened time axis
tl2 = times_1.shape[0]//80

ax2.plot((1e3)*times_1[1:tl2], unrestricted[:tl2-1], c=c7, label='Free')
ax2.plot((1e3)*times_1[1:tl2], msd_1[:tl2-1], c=c1, label=r'd = {:.0f} $\mu$m'.format((1e6)*cyl_radius_1*2))
# ax2.axhline(limit_1, c=c1)
ax2.plot((1e3)*times_1[1:tl2], msd_2[:tl2-1], c=c2, label=r'd = {:.0f} $\mu$m'.format((1e6)*cyl_radius_2*2))
ax2.axhline(limit_2, c=c2)
ax2.plot((1e3)*times_1[1:tl2], msd_3[:tl2-1], c=c3, label=r'd = {:.0f} $\mu$m'.format((1e6)*cyl_radius_3*2))
ax2.axhline(limit_3, c=c3)
ax2.plot((1e3)*times_1[1:tl2], msd_4[:tl2-1], c=c4, label=r'd = {:.0f} $\mu$m'.format((1e6)*cyl_radius_4*2))
ax2.axhline(limit_4, c=c4)
ax2.set_title('Zoom on the first {:.1f} ms'.format(((1e3)*times_1[1:tl2]).max()), size=24)

for axis in ['top','bottom','left','right']:
    ax2.spines[axis].set_linewidth(3)
    ax2.spines[axis].set_color('yellow')


# pl.show()
pl.savefig("Figure_6.png")


