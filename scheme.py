# sampling scheme utilities
import numpy as np


# convert ND into (N,1)
def _ravel2D(x):
	return np.ravel(x)[:,None]

# return all combination of parameters into (N,3)
def expand_scheme(list_G, list_DELTA, list_delta):
	tmp = np.meshgrid(list_G, list_DELTA, list_delta)
	return np.concatenate(list(map(_ravel2D, tmp)), axis=1)

# remove entries with delta > DELTA from scheme 
def remove_unphysical(scheme):
	return scheme[scheme[:,1] >= scheme[:,2]]

