import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from numba import jit
from joblib import Parallel, delayed
from scipy.signal import argrelextrema
v = 1
m = 0.5
mu1 = -50
W = 5
mu = -0.85
mu_m_divide_flatband=0.95

N_0_cutoff = 10000
num_M = 1000
num_T = 100
# Define the Energy function
@jit(nopython=True)
def Energy(n, B):
    Coefficient = np.array([1, -(mu1 + B * (n + 1/2) / m - 2 * m * pow(v, 2)),
                    -(pow(W, 2) + 2 * B * pow(v, 2) * (2 * n + 3/2) + 2 * m * mu1 * pow(v, 2)),
                    -4 * m * B * pow(v, 4) * (n + 1)])
    sortedroots=np.sort(np.roots(Coefficient))
    return sortedroots

# Define the GrandPotential function
@jit(nopython=True)
def GrandPotential(B,mu,T):
    # Ncutoff = int(np.floor(N_0_cutoff/B))
    Ncutoff = N_0_cutoff
    energy0=np.zeros(Ncutoff)
    energy1=np.zeros(Ncutoff)
    energy2=np.zeros(Ncutoff)
    Phi_finiteT=0
    for n in range(0,Ncutoff):
        energy=Energy(n,B)
        if (mu-energy[0])/T > 200:
            energy0_part1=(mu-energy[0])/T
        else:
            energy0_part1=np.log(1+np.exp((mu-energy[0])/T))
        if (mu+2*m*pow(v,2))/T > 200:
            energy0_part2=(mu+2*m*pow(v,2))/T
        else:
            energy0_part2=np.log(1+np.exp((mu+2*m*pow(v,2))/T))
        energy0[n] = energy0_part1 + energy0_part2
        
        if (mu-energy[1])/T > 200:
            energy1_part1=(mu-energy[1])/T
        else:
            energy1_part1=np.log(1+np.exp((mu-energy[1])/T))
        if (mu+2*m*pow(v,2))/T > 200:
            energy1_part2=(mu+2*m*pow(v,2))/T
        else:
            energy1_part2=np.log(1+np.exp((mu+2*m*pow(v,2))/T))
        energy1[n] = energy1_part1 + energy1_part2
        if (mu-energy[2])/T > 200:
            energy2[n]=(mu-energy[2])/T
        else:
            energy2[n]=np.log(1+np.exp((mu-energy[2])/T))    
    Phi_finiteT=-T*(np.sum(energy2)+np.sum(energy1)+np.sum(energy0))
    return Phi_finiteT*B
# Define the negative_power_polynomial_func function for polynomial fitting
def negative_power_polynomial_func(x, *coefficients):
    y = np.zeros_like(x, dtype=np.float64)
    for i, c in enumerate(coefficients):
        y += c * np.power(x, -i, dtype=np.float64)
    return y

# Define a function to calculate the oscillatory grand potential for a single temperature
def calculate_oscillatory_grand_potential(T, magnetic_field_inverse_range):
    grand_potential = np.zeros(len(magnetic_field_inverse_range))
    for j, magnetic_field_inverse in enumerate(magnetic_field_inverse_range):
        grand_potential[j] = GrandPotential(1/magnetic_field_inverse, mu, T)

    # Perform polynomial fitting for the grand potential
    degree = 2  # Define the degree of the polynomial (can be adjusted as needed)
    initial_guess = np.zeros(degree + 1)  # Provide an initial guess for the coefficients

    coefficients, _ = curve_fit(negative_power_polynomial_func, magnetic_field_inverse_range, grand_potential, p0=initial_guess)

    # Generate the background curve using the fitted polynomial coefficients
    background_curve = negative_power_polynomial_func(magnetic_field_inverse_range, *coefficients)

    # Subtract the background curve from the grand potential to obtain the oscillatory part
    oscillatory_part = grand_potential - background_curve

    return oscillatory_part
def calculate_oscillatory_M(T, magnetic_field_inverse_range):
    grand_potential = np.zeros(len(magnetic_field_inverse_range))
    for j, magnetic_field_inverse in enumerate(magnetic_field_inverse_range):
        grand_potential[j] = GrandPotential(1/magnetic_field_inverse, mu, T)
    M = np.gradient(grand_potential, magnetic_field_inverse_range)*magnetic_field_inverse_range**2
    
    # Perform polynomial fitting for the grand potential
    degree = 1  # Define the degree of the polynomial (can be adjusted as needed)
    initial_guess = np.zeros(degree + 1)  # Provide an initial guess for the coefficients

    coefficients, _ = curve_fit(negative_power_polynomial_func, magnetic_field_inverse_range, M, p0=initial_guess)

    # Generate the background curve using the fitted polynomial coefficients
    background_curve_M = negative_power_polynomial_func(magnetic_field_inverse_range, *coefficients)

    # Subtract the background curve from the grand potential to obtain the oscillatory part
    M_osci = M - background_curve_M

    return M_osci
# Calculate oscillatory grand potential for a range of magnetic fields and temperatures using parallel computing
magnetic_field_inverse_range = np.linspace(10, 10.02, num_M)
T_range = np.linspace(1e-7,1e-6 , num_T)

oscillatory_grand_M = Parallel(n_jobs=-1)(delayed(calculate_oscillatory_M)(T, magnetic_field_inverse_range) for T in tqdm(T_range, desc="Calculating M"))
oscillatory_grand_M = np.reshape(oscillatory_grand_M, (num_T,num_M))
max_index = argrelextrema(oscillatory_grand_M[1,:], np.greater)
min_index = argrelextrema(oscillatory_grand_M[1,:], np.less)
max_index = tuple(max_index[0].tolist())
min_index = tuple(min_index[0].tolist())
amplitude = oscillatory_grand_M[:,max_index]-oscillatory_grand_M[:,min_index]

plt.plot(T_range, amplitude)
plt.xlabel('Temperature (T)')
plt.ylabel('Amplitude of Oscillatory M')
plt.show()