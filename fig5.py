import numpy as np
import scipy.stats as ss
import pylab as pl
from string import ascii_lowercase

FILL_DIST = True
FILL_ALPHA = 0.2


from scipy.optimize import root_scalar

# analytical r_eff for a few distribution 
# r_eff for gamma distribution (k>0; shape, theta>0; scale); r in (0, +inf)
def reff_gamma(k, theta):
    return theta*((k+5)*(k+4)*(k+3)*(k+2))**0.25

# r_eff for normal distribution (mu; mean, sigma^2>0; variance); r in (-inf, +inf)
def reff_normal(mu, sigma):
    return ((mu**6+15*mu**4*sigma**2+45*mu**2*sigma**4+15*sigma**6)/(mu**2+sigma**2))**0.25

# r_eff for uniform distribution (a; low bound, b; upper bound); x in [a, b]
def reff_uniform(a, b):
    return ((3*(a**6+a**5*b+a**4*b**2+a**3*b**3+a**2*b**4+a*b**5+b**6))/(7*(a**2+a*b+b**2)))**0.25

# r_eff for chi distribution (k>0; d.o.f.); r in [0, +inf)
def reff_chi(k):
    return ((k+4)*(k+2))**0.25

# r_eff for chi2 distribution (k>0; d.o.f.); r in [0, +inf)
def reff_chi2(k):
    return 2*(((k/2)+5)*((k/2)+4)*((k/2)+3)*((k/2)+2))**0.25

# r_eff for exponential distribution (lambda>0; rate/inv scale); r in [0, +inf)
def reff_exp(lamb):
    return (360)**0.25 / lamb


def bracket_finder(value, func, startpoint=0):
    # finds (a,b) such that func(a) < value < func(b)
    # a,b > 0
    # assume monotone function
    # attempt to find if increasing or decreasing 
    if func(1) < func(10):
        k = 2.
    else:
        k = 0.5
    
    a = 2.**startpoint
    while func(a) >= value:
        a /= k
        
    b = 2**startpoint
    while func(b) <= value:
        b *= k
    return [a, b]

# storage
krange_gamma1 = []
thetarange_gamma1 = []
_reff_gamma1 = []

krange_gamma2 = []
thetarange_gamma2 = []
_reff_gamma2 = []

murange_normal1 = []
sigmarange_normal1 = []
_reff_normal1 = []

murange_normal2 = []
sigmarange_normal2 = []
_reff_normal2 = []

arange_uniform = []
brange_uniform = []
_reff_uniform = []

lambrange_exp = []
_reff_exp = []

krange_chi = []
_reff_chi = []



# diameters from liewald
_target_r_eff = [2.21, 1.13]


# optimize to find parameters
# all distribution are made univariate
for target_r_eff in _target_r_eff:

    # we want peak = 1
    # peak = (k-1)*theta
    peak = 1.
    def objfunc(x):
        return target_r_eff - reff_gamma((peak/x)+1, x)
    a,b = bracket_finder(0, objfunc)
    sol = root_scalar(objfunc, bracket=(a,b), xtol=1e-6)
    krange_gamma1.append((peak/sol.root)+1)
    thetarange_gamma1.append(sol.root)
    _reff_gamma1.append(reff_gamma((peak/sol.root)+1, sol.root))

    # we want peak = 0.5
    # peak = (k-1)*theta
    peak = 0.5
    def objfunc(x):
        return target_r_eff - reff_gamma((peak/x)+1, x)
    a,b = bracket_finder(0, objfunc)
    sol = root_scalar(objfunc, bracket=(a,b), xtol=1e-6)
    krange_gamma2.append((peak/sol.root)+1)
    thetarange_gamma2.append(sol.root)
    _reff_gamma2.append(reff_gamma((peak/sol.root)+1, sol.root))
    
    # normal(mu, 0.2)
	# print('normal1 {}'.format(k_gamma))
    sigma_normal1 = 0.2
    sigmarange_normal1.append(sigma_normal1)
    def objfunc(x):
        return target_r_eff - reff_normal(x, sigma_normal1)
    a,b = bracket_finder(0, objfunc)
    sol = root_scalar(objfunc, bracket=(a,b), xtol=1e-6)
    murange_normal1.append(sol.root)
    _reff_normal1.append(reff_normal(sol.root, sigma_normal1))


    # uniform(0, b)
	# print('uniform {}'.format(k_gamma))
    a_uniform = 0.0
    arange_uniform.append(a_uniform)
    def objfunc(x):
        return target_r_eff - reff_uniform(a_uniform, x)
    a,b = bracket_finder(0, objfunc)
    sol = root_scalar(objfunc, bracket=(a,b), xtol=1e-6)
    brange_uniform.append(sol.root)
    _reff_uniform.append(reff_uniform(a_uniform, sol.root))     
    
    # exp(lamb)
	# print('exp {}'.format(k_gamma))
    def objfunc(x):
        return target_r_eff - reff_exp(x)
    a,b = bracket_finder(0, objfunc)
    sol = root_scalar(objfunc, bracket=(a,b), xtol=1e-6)
    lambrange_exp.append(sol.root)
    _reff_exp.append(reff_exp(sol.root))


# Liewald diameter data
# data recovered from plots
data_left = np.genfromtxt('Liewald_2014_fig9_Human_brain_1_left.csv')
data_right = np.genfromtxt('Liewald_2014_fig9_Human_brain_1_right.csv')


# the bin center are known so we can correct
diams_left = np.round(data_left[:,0], 1)
diams_right = np.round(data_right[:,0], 1)
diams = [diams_left, diams_right]

# re-normalize to sum to 1
normCounts_left = data_left[:,1] / data_left[:,1].sum()
normCounts_right = data_right[:,1] / data_right[:,1].sum()
normCounts = [normCounts_left, normCounts_right]

delta_diam_left = 0.1
delta_diam_right = 0.1
delta_diam = [delta_diam_left, delta_diam_right]


# First set up the figure, the axis, and the plot element we want to animate
xs = [4, 2]
ys = [2, 4]


from matplotlib.ticker import FormatStrFormatter


dpi = 600
for i in range(len(_target_r_eff)):

    fig, ax = pl.subplots(figsize=(12, 7), dpi=dpi)

    qmin = 0.001 # min quantile
    qmax = 0.999 # max quantile

    ax.set_xlim((0, xs[i]))
    ax.set_ylim((0, ys[i]))

    line_gamma1, = ax.plot([], [], lw=3, label='gamma; peak = 1')
    line_gamma2, = ax.plot([], [], lw=3, label='gamma; peak = 0.5')
    line_normal1, = ax.plot([], [], lw=3, label='normal; sigma = 0.2')
    line_uniform, = ax.plot([], [], lw=3, label='uniform')
    line_exp, = ax.plot([], [], lw=3, label='exponential')
    reff_text = ax.text(xs[i]*0.75, ys[i]*0.5, '', fontsize=18)
    
    pl.legend(loc=1, fontsize=18)
    pl.xlabel('Diameters ($\mu$m)', fontsize=18)
    pl.xticks(fontsize=16)
    pl.yticks(fontsize=16)

    pl.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    pl.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # gamma1
    k_gamma1 = krange_gamma1[i]
    theta_gamma1 = thetarange_gamma1[i]
    rv = ss.gamma(k_gamma1, loc=0, scale=theta_gamma1) # define random variable
    rmin = rv.ppf(qmin) # find r corresponding to qmin
    rmax = rv.ppf(qmax) # find r corresponding to qmax
    rs = np.linspace(rmin, rmax, 1000, endpoint=True)
    pdf = rv.pdf(rs) # density at those r
    line_gamma1.set_data(rs, pdf)
    if FILL_DIST:
        pl.fill_between(rs, np.zeros_like(pdf), pdf, color=line_gamma1.get_color(), alpha=FILL_ALPHA)
    
    # gamma2
    k_gamma2 = krange_gamma2[i]
    theta_gamma2 = thetarange_gamma2[i]
    rv = ss.gamma(k_gamma2, loc=0, scale=theta_gamma2) # define random variable
    rmin = rv.ppf(qmin) # find r corresponding to qmin
    rmax = rv.ppf(qmax) # find r corresponding to qmax
    rs = np.linspace(rmin, rmax, 1000, endpoint=True)
    pdf = rv.pdf(rs) # density at those r
    line_gamma2.set_data(rs, pdf)
    if FILL_DIST:
        pl.fill_between(rs, np.zeros_like(pdf), pdf, color=line_gamma2.get_color(), alpha=FILL_ALPHA)
    
    # normal1
    mu_normal1 = murange_normal1[i]
    sigma_normal1 = sigmarange_normal1[i]
    rv = ss.norm(loc=mu_normal1, scale=sigma_normal1) # define random variable
    rmin = rv.ppf(qmin) # find r corresponding to qmin
    rmax = rv.ppf(qmax) # find r corresponding to qmax
    rs = np.linspace(rmin, rmax, 1000, endpoint=True)
    pdf = rv.pdf(rs) # density at those r
    line_normal1.set_data(rs, pdf)
    if FILL_DIST:
        pl.fill_between(rs, np.zeros_like(pdf), pdf, color=line_normal1.get_color(), alpha=FILL_ALPHA)

    # uniform
    a_uniform = arange_uniform[i]
    b_uniform = brange_uniform[i]
    rv = ss.uniform(loc=a_uniform, scale=b_uniform-a_uniform) # define random variable
    rmin = rv.ppf(qmin) # find r corresponding to qmin
    rmax = rv.ppf(qmax) # find r corresponding to qmax
    rs = np.linspace(rmin, rmax, 1000, endpoint=True)
    pdf = rv.pdf(rs) # density at those r
    line_uniform.set_data(rs, pdf)
    if FILL_DIST:
        pl.fill_between(rs, np.zeros_like(pdf), pdf, color=line_uniform.get_color(), alpha=FILL_ALPHA)
    
    # uniform
    lamb_exp = lambrange_exp[i]
    rv = ss.expon(loc=0, scale=1/lamb_exp) # define random variable
    rmin = rv.ppf(qmin) # find r corresponding to qmin
    rmax = rv.ppf(qmax) # find r corresponding to qmax
    rs = np.linspace(rmin, rmax, 1000, endpoint=True)
    pdf = rv.pdf(rs) # density at those r
    line_exp.set_data(rs, pdf)
    if FILL_DIST:
        pl.fill_between(rs, np.zeros_like(pdf), pdf, color=line_exp.get_color(), alpha=FILL_ALPHA)
    
    reff_text.set_text('$d_{{eff}}$ = {:.2f} $\mu$m'.format(_target_r_eff[i]))

    for j in range(diams[i].shape[0]):
        pl.bar(diams[i][j], normCounts[i][j]/delta_diam[i], delta_diam[i], color='k')
 
    pl.savefig("Figure_5{1:}.png".format(ascii_lowercase[i]))

# pl.show()



