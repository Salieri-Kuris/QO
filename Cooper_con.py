import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
from numba import jit
m=1
gamma_d_W=0.05
W=1
mu_m_d_W=0.5*(5+1-np.sqrt(16+0.05**2))
Ncutoff=10000
gamma=W*gamma_d_W
mu_d_W=np.array([1.5,0.5,0.9995,1])
mu_m=W*mu_m_d_W
mu=W*mu_d_W
@jit(nopython=True)
def Energy1(n, B):   
    return 0.5*(B*(n+0.5)+W+np.sqrt((B*(n+0.5)-W)**2+gamma**2))
@jit(nopython=True)
def Energy2(n, B):   
    return 0.5*(B*(n+0.5)+W-np.sqrt((B*(n+0.5)-W)**2+gamma**2))
@jit(nopython=True)
def GrandPotential(B,mu):
    Phi=0
    for n in range(0,Ncutoff):
        if Energy1(n,B)<mu:
            Phi+=Energy1(n,B)-mu
        if Energy2(n,B)<mu:
            Phi+=Energy2(n,B)-mu
    return Phi*B
M=[]
def negative_power_polynomial_func(x, *coefficients):
    y = np.zeros_like(x, dtype=np.float64)
    for i, c in enumerate(coefficients):
        y += c * np.power(x, -i, dtype=np.float64)
    return y
for Mu in mu:
    magnetic_field_inverse_range=np.arange(2,12, 0.01)
    grand_potential = np.array(Parallel(n_jobs=-1)(delayed(GrandPotential)(1/b, Mu) for b in tqdm(magnetic_field_inverse_range, desc="Calculating Grand Potential")))
    degree = 2  # Define the degree of the polynomial (can be adjusted as needed)
    initial_guess = np.zeros(degree + 1)  # Provide an initial guess for the coefficients

    coefficients, _ = curve_fit(negative_power_polynomial_func, magnetic_field_inverse_range, grand_potential, p0=initial_guess)

    # Generate the background curve using the fitted polynomial coefficients
    background_curve = negative_power_polynomial_func(magnetic_field_inverse_range, *coefficients)

    # Subtract the background curve from the grand potential to obtain the oscillatory part
    oscillatory_part = grand_potential - background_curve

    M.append(np.gradient(oscillatory_part, magnetic_field_inverse_range)*magnetic_field_inverse_range**2)
    

# plt.plot(magnetic_field_inverse_range ,grand_potential)
# plt.xlabel("1/B")
# plt.title("phi")
# plt.show()

# plt.plot(magnetic_field_inverse_range ,oscillatory_part)
# plt.xlabel("1/B")
# plt.title("phi_osi")
# plt.show()


for i in range(0,3):
    plt.plot(magnetic_field_inverse_range ,M[i])
plt.xlabel("1/B")
plt.title("M")
plt.show()