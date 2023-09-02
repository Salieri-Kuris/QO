import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from numba import jit
from scipy import stats
v = 1
m = 0.5
mu1 = -50
W = 5
mu = -0.85
N_0_cutoff = 10000
finite = 0
zero = 1
correctness=0
T_finite=1e-6
B_inverse_step = 2e-5
propotion = 1.0005
mu_cutoff_0 = -propotion * 2*m*v**2
mu_cutoff_1 = -1/propotion * 2*m*v**2
magnetic_field_inverse_range = np.arange(1, 1.02, B_inverse_step)
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
@jit(nopython=True)
def GrandPotentialZeroTWithPV(B, mu):
    # Ncutoff = int(np.floor(N_0_cutoff/B))
    Ncutoff = N_0_cutoff
    Phi = 0
    for n in range(Ncutoff):
        energy = Energy(n, B)
        if mu > -2*m*v**2:
            if energy[0] < mu:
                Phi += energy[0] + 2*m*v**2
            else:
                Phi += mu + 2*m*v**2
            if energy[1] < mu:
                Phi += energy[1] + 2*m*v**2
            else:
                Phi += mu + 2*m*v**2 
        else:
            if energy[0] < mu:
                Phi += energy[0] - mu
        if energy[2] < mu:
            Phi += energy[2] - mu

    return Phi * B
@jit(nopython=True)
def GrandPotentialZeroT_Cooper(B, mu):
    Ncutoff = N_0_cutoff
    Phi = 0
    for n in range(Ncutoff):
        energy = Energy(n, B)
        if energy[0] < mu_cutoff_0:
            Phi += energy[0] - mu
        if energy[1] < mu_cutoff_1:
            Phi += energy[0] - mu        
        if energy[2] < mu:
            Phi += energy[2] - mu
    return Phi * B
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

@jit(nopython=True)
def LL_Counting(B, mu):
    Num = 0
    for n in range(0, N_0_cutoff):
        Num += np.sum(Energy(n, B) < mu)
    return Num

num_LL_below_mu = np.array(Parallel(n_jobs=-1)(delayed(LL_Counting)(1/b, mu) for b in tqdm(magnetic_field_inverse_range, desc="Calculating Number of LL below mu")))
plt.plot(magnetic_field_inverse_range, num_LL_below_mu)
plt.xlabel("1/B")
plt.title("Number of LL below mu")
plt.show()
if zero == 1:
    T = 0
    grand_potential_zeroT = np.array(Parallel(n_jobs=-1)(delayed(GrandPotentialZeroT_Cooper)(1/b, mu) for b in tqdm(magnetic_field_inverse_range, desc="Calculating Grand Potential")))
    # Calculate the z-score for each value in grand_potential


    degree = 2  # Define the degree of the polynomial (can be adjusted as needed)
    initial_guess = np.zeros(degree + 1)  # Provide an initial guess for the coefficients

    coefficients, _ = curve_fit(negative_power_polynomial_func, magnetic_field_inverse_range, grand_potential_zeroT, p0=initial_guess)

    # Generate the background curve using the fitted polynomial coefficients
    background_curve = negative_power_polynomial_func(magnetic_field_inverse_range, *coefficients)

    # Subtract the background curve from the grand potential to obtain the oscillatory part
    oscillatory_part = grand_potential_zeroT - background_curve
    # M_osci1 = np.gradient(oscillatory_part, magnetic_field_inverse_range)*magnetic_field_inverse_range**2
    M = np.gradient(grand_potential_zeroT, magnetic_field_inverse_range)*magnetic_field_inverse_range**2
    z_scores_M = stats.zscore(M)
    threshold = 1
    M_correct = M[abs(z_scores_M) < threshold]
    magnetic_field_inverse_range_correct = magnetic_field_inverse_range[abs(z_scores_M) < threshold]
    # z_scores_Mcorrect = stats.zscore(M_correct)
    # threshold = 2
    # M_correct = M_correct[abs(z_scores_Mcorrect) < threshold]
    # magnetic_field_inverse_range_correct = magnetic_field_inverse_range_correct[abs(z_scores_Mcorrect) < threshold]
    degree = 1  # Define the degree of the polynomial (can be adjusted as needed)
    initial_guess = np.zeros(degree + 1)  # Provide an initial guess for the coefficients

    coefficients, _ = curve_fit(negative_power_polynomial_func, magnetic_field_inverse_range, M, p0=initial_guess)

    # Generate the background curve using the fitted polynomial coefficients
    background_curve_M = negative_power_polynomial_func(magnetic_field_inverse_range, *coefficients)

    # Subtract the background curve from the grand potential to obtain the oscillatory part
    M_osci1 = M - background_curve_M

    coefficients, _ = curve_fit(negative_power_polynomial_func, magnetic_field_inverse_range_correct, M_correct, p0=initial_guess)

    # Generate the background curve using the fitted polynomial coefficients
    background_curve_M = negative_power_polynomial_func(magnetic_field_inverse_range_correct, *coefficients)

    # Subtract the background curve from the grand potential to obtain the oscillatory part
    M_osci2 = M_correct - background_curve_M



    plt.plot(magnetic_field_inverse_range, grand_potential_zeroT)
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

    if correctness == 1:
        plt.plot(magnetic_field_inverse_range_correct, M_correct)
        plt.xlabel("1/B")
        plt.title("M corrected")
        plt.show()
        plt.plot(magnetic_field_inverse_range_correct, M_osci2)
        plt.xlabel("1/B")
        plt.title("osillatory part of M correct")
        plt.show()

    plt.plot(magnetic_field_inverse_range, M_osci1)
    plt.xlabel("1/B")
    plt.title("osillatory part of M")
    plt.show()




if finite == 1:
    T=T_finite
    grand_potential_finiteT = np.array(Parallel(n_jobs=-1)(delayed(GrandPotential)(1/b, mu, T) for b in tqdm(magnetic_field_inverse_range, desc="Calculating Grand Potential")))

    degree = 2  # Define the degree of the polynomial (can be adjusted as needed)
    initial_guess = np.zeros(degree + 1)  # Provide an initial guess for the coefficients

    coefficients, _ = curve_fit(negative_power_polynomial_func, magnetic_field_inverse_range, grand_potential_finiteT, p0=initial_guess)

    # Generate the background curve using the fitted polynomial coefficients
    background_curve = negative_power_polynomial_func(magnetic_field_inverse_range, *coefficients)

    # Subtract the background curve from the grand potential to obtain the oscillatory part
    oscillatory_part = grand_potential_finiteT - background_curve
    # M_osci1 = np.gradient(oscillatory_part, magnetic_field_inverse_range)*magnetic_field_inverse_range**2
    M = np.gradient(grand_potential_finiteT, magnetic_field_inverse_range)*magnetic_field_inverse_range**2
    z_scores_M = stats.zscore(M)
    threshold = 2
    M_correct = M[abs(z_scores_M) < threshold]

    magnetic_field_inverse_range_correct = magnetic_field_inverse_range[abs(z_scores_M) < threshold]
    degree = 1  # Define the degree of the polynomial (can be adjusted as needed)
    initial_guess = np.zeros(degree + 1)  # Provide an initial guess for the coefficients

    coefficients, _ = curve_fit(negative_power_polynomial_func, magnetic_field_inverse_range, M, p0=initial_guess)

    # Generate the background curve using the fitted polynomial coefficients
    background_curve_M = negative_power_polynomial_func(magnetic_field_inverse_range, *coefficients)

    # Subtract the background curve from the grand potential to obtain the oscillatory part
    M_osci1 = M - background_curve_M

    coefficients, _ = curve_fit(negative_power_polynomial_func, magnetic_field_inverse_range_correct, M_correct, p0=initial_guess)

    # Generate the background curve using the fitted polynomial coefficients
    background_curve_M = negative_power_polynomial_func(magnetic_field_inverse_range_correct, *coefficients)

    # Subtract the background curve from the grand potential to obtain the oscillatory part
    M_osci2 = M_correct - background_curve_M
    plt.plot(magnetic_field_inverse_range, grand_potential_finiteT)
    plt.xlabel("1/B")
    plt.title(f"total phi corrected,mu={mu},1/B step={B_inverse_step},T={T}")
    plt.show()

    plt.plot(magnetic_field_inverse_range, oscillatory_part)
    plt.xlabel("1/B")
    plt.title("oscillatory_part of grand potential")
    plt.show()

    plt.plot(magnetic_field_inverse_range, M)
    plt.xlabel("1/B")
    plt.title("M")
    plt.show()

    if correctness == 1:
        plt.plot(magnetic_field_inverse_range_correct, M_correct)
        plt.xlabel("1/B")
        plt.title("M corrected")
        plt.show()
        plt.plot(magnetic_field_inverse_range_correct, M_osci2)
        plt.xlabel("1/B")
        plt.title("osillatory part of M correct")
        plt.show()

    plt.plot(magnetic_field_inverse_range, M_osci1)
    plt.xlabel("1/B")
    plt.title("osillatory part of M")
    plt.show()
