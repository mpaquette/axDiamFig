# Van Gelderen formula for diffusion in cylinders
import numpy as np
from scipy.special import jnp_zeros



#  T^-1 s^-1
gamma = 42.515e6 * 2*np.pi


def vangelderen_cylinder_perp_ln_terms(D, R, scheme, m_max=10):
	# returns the scaling factors and the list of each summation component for ln(M(DELTA,delta,G)/M(0))
	# D:= free diffusivity in m^2 s^-1
	# R:= cylinder radius in m
	# scheme is (N, 3) and contains all the (G, DELTA, delta)
		# G:= gradient magnitude in T m^-1
		# DELTA:= gradient separation in s
		# delta:= gradient width s

	am_R = jnp_zeros(1,m_max)[:,None]
	am = am_R / R
	am2 = am**2

	# unpack for clarity
	G = scheme[:,0]
	DELTA = scheme[:,1]
	delta = scheme[:,2]

	# multiplicative factor for each scheme point
	fac = -2*gamma**2*G**2/D**2

	# summation terms until m_max, for each scheme point
	comp = (1/(am**6*(am_R**2-1))) * (2*D*am2*delta - 2 + 2*np.exp(-D*am2*delta) + 2*np.exp(-D*am2*DELTA) - np.exp(-D*am2*(DELTA-delta)) - np.exp(-D*am2*(DELTA+delta)))

	return fac, comp


def vangelderen_cylinder_perp(D, R, scheme, m_max=10):
	# get the parts
	fac, comp = vangelderen_cylinder_perp_ln_terms(D, R, scheme, m_max)
	lnS = fac*np.sum(comp, axis=0)
	return np.exp(lnS)



def vangelderen_cylinder_perp_ln_terms_TAYLOR(D, R, scheme, m_max=10):
	# not going to bother changing the description, it's the same thing but every exponential inside the sum as been taylored (exp(-x) = 1 + x)
	# by doing so we can take out everything other then the bessel roots from the summation
	# DISCLAIMER, this computation is weirdly off by an exact factor of 2 from Neuman

	# returns the scaling factors and the list of each summation component for ln(M(DELTA,delta,G)/M(0))
	# D:= free diffusivity in m^2 s^-1
	# R:= cylinder radius in m
	# scheme is (N, 3) and contains all the (G, DELTA, delta)
		# G:= gradient magnitude in T m^-1
		# DELTA:= gradient separation in s
		# delta:= gradient width s

	xm = jnp_zeros(1,m_max)[:,None]

	# unpack for clarity
	G = scheme[:,0]
	DELTA = scheme[:,1]
	delta = scheme[:,2]

	# multiplicative factor for each scheme point
	fac = -8*gamma**2*G**2*R**4/D

	# summation terms until m_max, for each scheme point
	comp = (1/(xm**4*(xm**2-1)))

	return fac, comp


def vangelderen_cylinder_perp_TAYLOR(D, R, scheme, m_max=10):
	# see vangelderen_cylinder_perp_ln_terms_TAYLOR
	
	# get the parts
	fac, comp = vangelderen_cylinder_perp_ln_terms_TAYLOR(D, R, scheme, m_max)
	lnS = fac*np.sum(comp, axis=0)
	return np.exp(lnS)





