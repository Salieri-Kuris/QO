import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from numba import jit, cuda
import cupy as cp
from scipy import stats
import operator
v = 1
m = 0.5
mu1 = -50
W = 5
mu = -0.6
N_0_cutoff = float(10000)
T = 0

def find_peaks_1d(data, height=None, distance=None):    
    peaks, _ = find_peaks(data, height=height, distance=distance)
    return peaks

# Define the Energy function
@jit(nopython=True)
def Energy(n, B):
    Coefficient = np.array([1, -(mu1 + B * (n + 1/2) / m - 2 * m * pow(v, 2)),
                    -(pow(W, 2) + 2 * B * pow(v, 2) * (2 * n + 3/2) + 2 * m * mu1 * pow(v, 2)),
                    -4 * m * B * pow(v, 4) * (n + 1)])
    sortedroots=np.sort(np.roots(Coefficient))
    return sortedroots

@cuda.jit(device=True)
def GrandPotentialZeroTWithPV_kernel(B, mu):
    Ncutoff = operator.floordiv(N_0_cutoff, B)
    Phi = 0
    for n in range(Ncutoff):
        energy = Energy(n, B)
        if mu > -2*m*v**2:
            if energy[0] < mu:
                Phi += energy[0] + 2
            else:
                Phi += mu + 2
            if energy[1] < mu:
                Phi += energy[1] + 2
            else:
                Phi += mu + 2 
        else:
            if energy[0] < mu:
                Phi += energy[0] - mu
        if energy[2] < mu:
            Phi += energy[2] - mu

    return Phi * B

@jit(nopython=True)
def GrandPotentialZeroTWithPV(B, mu):
    return GrandPotentialZeroTWithPV_kernel(B, mu)

@cuda.jit(device=True)
def GrandPotential_kernel(B, mu, T, energy1, energy2, energy3):
    Ncutoff = operator.floordiv(N_0_cutoff, B)
    for n in range(0,Ncutoff):
        energy=Energy(n,B)
        if (mu-energy[0])/T > 150:
            energy1[n]=(mu-energy[0])/T-np.log(1+np.exp((mu+2*m*pow(v,2))/T))
            energy2[n]=np.log(1+np.exp((mu-energy[1])/T))-np.log(1+np.exp((mu+2*m*pow(v,2))/T))
            energy3[n]=np.log(1+np.exp((mu-energy[2])/T))
        else:
            energy1[n]=np.log(1+np.exp((mu-energy[0])/T))-np.log(1+np.exp((mu+2*m*pow(v,2))/T))
            energy2[n]=np.log(1+np.exp((mu-energy[1])/T))-np.log(1+np.exp((mu+2*m*pow(v,2))/T))
            energy3[n]=np.log(1+np.exp((mu-energy[2])/T))

    Phi=-T*(cp.sum(energy2)+cp.sum(energy1)+cp.sum(energy3))
    return Phi*B

@jit(nopython=True)
def GrandPotential(B,mu,T):
    energy1 = cp.zeros(N_0_cutoff)
    energy2 = cp.zeros(N_0_cutoff)
    energy3 = cp.zeros(N_0_cutoff)
    magnetic_field_inverse_range_gpu = cp.asarray(magnetic_field_inverse_range)
    grand_potential_gpu = cp.zeros_like(magnetic_field_inverse_range_gpu)
    for i in range(len(magnetic_field_inverse_range)):
        grand_potential_gpu[i] = GrandPotential_kernel(1/magnetic_field_inverse_range_gpu[i], mu, T, energy1, energy2, energy3)
    return cp.asnumpy(grand_potential_gpu)

# Define the negative_power_polynomial_func function for polynomial fitting
def negative_power_polynomial_func(x, *coefficients):
    y = np.zeros_like(x, dtype=np.float64)
    for i, c in enumerate(coefficients):
        y += c * np.power(x, -i, dtype=np.float64)
    return y

@jit(nopython=True)
def LL_Counting(B, mu):
    Num = 0
    for n in range(0, N_0_cutoff):
        Num += np.sum(Energy(n, B) < mu)
    return Num

B_inverse_step = 0.0001
magnetic_field_inverse_range = np.arange(10, 11, B_inverse_step)

# grand_potential = GrandPotential(magnetic_field_inverse_range, mu, T)
grand_potential = GrandPotentialZeroTWithPV(magnetic_field_inverse_range, mu)
# Calculate the z-score for each value in grand_potential
z_scores = stats.zscore(grand_potential)

# Set a threshold for the z-score
threshold = 3

# Remove the values that have a z-score larger than the threshold
grand_potential_corrected = grand_potential[abs(z_scores) < threshold]
magnetic_field_inverse_range_corrected = magnetic_field_inverse_range[abs(z_scores) < threshold]

degree = 2  # Define the degree of the polynomial (can be adjusted as needed)
initial_guess = np.zeros(degree + 1)  # Provide an initial guess for the coefficients

coefficients, _ = curve_fit(negative_power_polynomial_func, magnetic_field_inverse_range_corrected, grand_potential_corrected, p0=initial_guess)

# Generate the background curve using the fitted polynomial coefficients
background_curve = negative_power_polynomial_func(magnetic_field_inverse_range, *coefficients)

# Subtract the background curve from the grand potential to obtain the oscillatory part
oscillatory_part = grand_potential - background_curve
M_osci1 = np.gradient(oscillatory_part, magnetic_field_inverse_range)*magnetic_field_inverse_range**2
M = np.gradient(grand_potential, magnetic_field_inverse_range)*magnetic_field_inverse_range**2

degree = 1  # Define the degree of the polynomial (can be adjusted as needed)
initial_guess = np.zeros(degree + 1)  # Provide an initial guess for the coefficients

coefficients, _ = curve_fit(negative_power_polynomial_func, magnetic_field_inverse_range, M, p0=initial_guess)

# Generate the background curve using the fitted polynomial coefficients
background_curve_M = negative_power_polynomial_func(magnetic_field_inverse_range, *coefficients)

# Subtract the background curve from the grand potential to obtain the oscillatory part
M_osci2 = M - background_curve_M

peaks1 = find_peaks_1d(M_osci1)
peaks2 = find_peaks_1d(M_osci2)

num_LL_below_mu = np.array(Parallel(n_jobs=-1)(delayed(LL_Counting)(1/b, mu) for b in tqdm(magnetic_field_inverse_range, desc="Calculating Number of LL below mu")))

plt.plot(magnetic_field_inverse_range, grand_potential)
plt.xlabel("1/B")
plt.title(f"total phi,mu={mu},1/B step={B_inverse_step},T={T}")
plt.show()

plt.plot(magnetic_field_inverse_range, oscillatory_part)
plt.xlabel("1/B")
plt.title("oscillatory_part of grand potential")
plt.show()

plt.plot(magnetic_field_inverse_range, M)
plt.xlabel("1/B")
plt.title("M")
plt.show()

plt.plot(magnetic_field_inverse_range, M_osci1)
# plt.scatter(magnetic_field_inverse_range[peaks1], M_osci1[peaks1], color='red', marker='o')
plt.xlabel("1/B")
plt.title("osillatory part 1 of M")
plt.show()

plt.plot(magnetic_field_inverse_range, M_osci2)
# plt.scatter(magnetic_field_inverse_range[peaks2], M_osci2[peaks2], color='red', marker='o')
plt.xlabel("1/B")
plt.title("osillatory part 2 of M")
plt.show()

plt.plot(magnetic_field_inverse_range, num_LL_below_mu)
plt.xlabel("1/B")
plt.title("Number of LL below mu")
plt.show()
