import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
v = 1
m = 0.5
mu1 = -50
W = 5
mu = -0.85
N_0_cutoff = 10000
T = 0
def negative_power_polynomial_func(x, *coefficients):
    y = np.zeros_like(x, dtype=np.float64)
    for i, c in enumerate(coefficients):
        y += c * np.power(x, -i, dtype=np.float64)
    return y
def Energy(n, B):
    Coefficient = [1, -(mu1 + B * (n + 1/2) / m - 2 * m * pow(v, 2)),
                    -(pow(W, 2) + 2 * B * pow(v, 2) * (2 * n + 3/2) + 2 * m * mu1 * pow(v, 2)),
                    -4 * m * B * pow(v, 4) * (n + 1)]
    sortedroots=np.sort(np.roots(Coefficient))
    return sortedroots


def find_peaks_1d(data, height=None, distance=None):

    peaks, _ = find_peaks(data, height=height, distance=distance)
    return peaks
def GrandPotentialZeroTWithPV(B, mu):
    Ncutoff = int(np.floor(N_0_cutoff/B))
    Phi = 0
    for n in range(Ncutoff):
        energy=Energy(n, B)
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
        if energy[2]<mu:
            Phi+=energy[2]-mu

    return Phi * B

def LL_Counting(B,mu):
    Num=0
    for n in range(0,N_0_cutoff):
        Num += np.sum(Energy(n, B)<mu)
    return Num
    

def GrandPotential(B,mu,T):
    Ncutoff = int(np.floor(N_0_cutoff/B))
    energy1=np.zeros(Ncutoff)
    energy2=np.zeros(Ncutoff)
    energy3=np.zeros(Ncutoff)
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
    Phi=-T*(np.sum(energy2)+np.sum(energy1)+np.sum(energy3))
    return Phi*B
B_inverse_step=2e-5
magnetic_field_inverse_range=np.arange(10,10.02 , B_inverse_step)
grand_potential = np.array(Parallel(n_jobs=-1)(delayed(GrandPotentialZeroTWithPV)(1/b, mu) for b in tqdm(magnetic_field_inverse_range, desc="Calculating Grand Potential")))
# grand_potential = np.array(Parallel(n_jobs=-1)(delayed(GrandPotential)(1/b, mu, T) for b in tqdm(magnetic_field_inverse_range, desc="Calculating Grand Potential")))
degree = 2  # Define the degree of the polynomial (can be adjusted as needed)
initial_guess = np.zeros(degree + 1)  # Provide an initial guess for the coefficients

coefficients, _ = curve_fit(negative_power_polynomial_func, magnetic_field_inverse_range, grand_potential, p0=initial_guess)

# Generate the background curve using the fitted polynomial coefficients
background_curve = negative_power_polynomial_func(magnetic_field_inverse_range, *coefficients)

# Subtract the background curve from the grand potential to obtain the oscillatory part
oscillatory_part = grand_potential - background_curve

M = np.gradient(grand_potential, magnetic_field_inverse_range)*magnetic_field_inverse_range**2
M_osci1 = np.gradient(oscillatory_part, magnetic_field_inverse_range)*magnetic_field_inverse_range**2

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

plt.plot(magnetic_field_inverse_range ,grand_potential)
plt.xlabel("1/B")
# plt.title("total phi," + "mu=" + str(mu) +",1/B step = " + str(B_inverse_step) + ",T=" + str(T))
plt.title(f"total phi,mu= {mu},1/B step={B_inverse_step},T={T}")
plt.show()

plt.plot(magnetic_field_inverse_range ,oscillatory_part)
plt.xlabel("1/B")
plt.title("oscillatory_part of grand potential")
plt.show()

plt.plot(magnetic_field_inverse_range ,M)
plt.xlabel("1/B")
plt.title("M")
plt.show()

plt.plot(magnetic_field_inverse_range ,M_osci1)
# plt.scatter(magnetic_field_inverse_range[peaks1], M_osci1[peaks1], color='red', marker='o')
plt.xlabel("1/B")
plt.title("osillatory part 1 of M")
plt.show()

plt.plot(magnetic_field_inverse_range ,M_osci2)
# plt.scatter(magnetic_field_inverse_range[peaks2], M_osci2[peaks2], color='red', marker='o')
plt.xlabel("1/B")
plt.title("osillatory part 2 of M")
plt.show()

plt.plot(magnetic_field_inverse_range ,num_LL_below_mu)
plt.xlabel("1/B")
plt.title("Number of LL below mu")
plt.show()
