#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl
from string import ascii_lowercase

from scipy.stats import norm, bayes_mvs, gaussian_kde

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}\usepackage{{amsfonts}}'


#  T^-1 s^-1
gamma = 42.515e6 * 2*np.pi

def nilsson_diameter(sigma_bar, D_0, delta, G):
	# d_min =  ((768 sigma_bar D_0) / (7 gamma^2 delta G^2))^(1/4)
	# assume delta = DELTA
	# valid (expanded) around low diffusivity (aka low d_min)
	return ((768 * sigma_bar * D_0) / (7. * gamma**2 * delta * G**2))**(1/4.)

def nilsson_dS(d, D_0, delta, G):
	# dS/S_0 =  (7 gamma^2 delta G^2 d^4) / (768 D_0)
	# assume delta = DELTA
	# valid (expanded) around low diffusivity (aka low d_min)
	return ((7. * gamma**2 * delta * G**2) / (768 * D_0)) * d**4


# invivo diffusivity
D0 = 2e-9
# exvivo diffusivity
# D0 = 0.66e-9

# most sensitive Connectom-like scheme
# grad (T/m), DELTA (s), delta (s) 
scheme_connectom = np.array([[0.3, 40e-3, 40e-3]])

# data SNR at B0
SNRs = [30.0, 300.0]

for isnr in range(len(SNRs)):
	SNR = SNRs[isnr]

	# significance level
	alphas = np.array([0.05])
	# compute sigma_bar for the diameter limit formula
	sigmabs = norm().ppf(1-alphas) / SNR

	d_mins = nilsson_diameter(sigmabs, D0, scheme_connectom[0,2], scheme_connectom[0,0])

	# diameters in m
	diams = np.arange(0.1, 5.1, 0.05)*1e-6

	# number of noise trial
	Ntrial = 10000

	np.random.seed(0) 
	fit_data = np.zeros((len(diams), Ntrial), dtype=np.complex) # when gaussian noise bring signal > 1 (i.e. b0), you get imaginary diameter (to be disarded)

	for idiam, diam in enumerate(diams):
		noiseless_signal = nilsson_dS(diam, D0, scheme_connectom[0,2], scheme_connectom[0,0])
		noiseless_fit = nilsson_diameter(noiseless_signal, D0, scheme_connectom[0,2], scheme_connectom[0,0])
		print('D = {:.2f}   Fit = {:.2f}'.format(1e6*diam, 1e6*noiseless_fit))
		for itrial in range(Ntrial):
			noise = (1/float(SNR))*np.random.randn()
			noisy_fit = nilsson_diameter(noiseless_signal+noise, D0, complex(scheme_connectom[0,2]), complex(scheme_connectom[0,0]))
			fit_data[idiam, itrial] = noisy_fit

	rejectCount = (np.abs(np.imag(fit_data)) > 0).sum(axis=1)
	fit_data = np.real(fit_data)

	### computing the mean/std for each radii 
	alpha_B = 0.95
	bcv = []
	for i in range(fit_data.shape[0]):
		bcv.append(bayes_mvs(fit_data[i], alpha_B))


	bayes_mean_stat = np.array([s[0].statistic for s in bcv])
	bayes_mean_stat_min = np.array([s[0].minmax[0] for s in bcv])
	bayes_mean_stat_max = np.array([s[0].minmax[1] for s in bcv])

	bayes_std_stat = np.array([s[2].statistic for s in bcv])
	bayes_std_stat_min = np.array([s[2].minmax[0] for s in bcv])
	bayes_std_stat_max = np.array([s[2].minmax[1] for s in bcv])


	## estimating data median to get a left and right side X% interval
	interval = 0.8
	peak_diams_mean = np.zeros(fit_data.shape[0])
	lower_diams_mean = np.zeros(fit_data.shape[0])
	upper_diams_mean = np.zeros(fit_data.shape[0])

	for i in range(fit_data.shape[0]):

		peak_diams_mean[i] = bayes_mean_stat[i]
		lower_diams_mean[i] = np.quantile(fit_data[i][fit_data[i]<=bayes_mean_stat[i]], 1-interval)
		upper_diams_mean[i] = np.quantile(fit_data[i][fit_data[i]>=bayes_mean_stat[i]], interval)

	dpi = 100
	pl.figure(figsize=(10,10), dpi=dpi)

	jitter_intensity = 0.5
	step = (diams[1:] - diams[:-1]).mean()
	jitter = (0.5-np.random.rand(Ntrial*diams.shape[0]))*step*jitter_intensity
	pl.scatter((np.repeat(diams, Ntrial)+jitter)*1e6, fit_data.ravel()*1e6, color='red', alpha=0.01, edgecolors="none")


	pl.plot(diams*1e6, diams*1e6, color='black', linestyle='--')

	pl.plot(diams*1e6, bayes_mean_stat*1e6, color='limegreen', linewidth=4, label='Mean fitted diameter', zorder=3) # zorder thinkering so that we can plot mean OVER the CI

	color1='blue'

	idx_trunc = max(np.where(lower_diams_mean > 0)[0][0] - 1, 0)
	pl.plot(diams[idx_trunc:]*1e6, lower_diams_mean[idx_trunc:]*1e6, color=color1, linewidth=4, label=r'{:.0f}\% Confidence Interval'.format(100*interval))
	pl.plot(diams*1e6, upper_diams_mean*1e6, color=color1, linewidth=4)


	for d_i, d_min in enumerate(d_mins):
		pl.axvline(d_min*1e6, color='orange', label=r'$d_{{\min}}$ = {:.2f} $\mu$m ({:.1f} \% decay with $\alpha$ = {})'.format(d_min*1e6, 100*sigmabs[d_i], alphas[d_i]), linewidth=3, zorder=1)

	pl.xlim([0, 1.05*np.max(diams)*1e6])
	pl.ylim([0, 1.05*np.max(diams)*1e6])

	pl.xlabel(r'True diameters ($\mu$m)', fontsize=24)
	pl.ylabel(r'Fitted diameters ($\mu$m)', fontsize=24)

	pl.xticks(fontsize=16)
	pl.yticks(fontsize=16)

	pl.gca().set_aspect('equal')

	pl.legend(loc=2, fontsize=22)

	pl.text(3.5, 1.0, 'SNR = {:.0f}'.format(SNR), fontsize=22)

	pl.savefig("Figure_1{}.png".format(ascii_lowercase[isnr]))

# pl.show()
