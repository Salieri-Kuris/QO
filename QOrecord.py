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
T = 1e-5
B=0.2
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
    Ncutoff = int(np.floor(N_0_cutoff/B))
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

# Define the GrandPotential function
# @jit(nopython=True)
# def GrandPotential(B,mu,T):
#     Ncutoff = int(np.floor(N_0_cutoff/B))
#     energy1=np.zeros(Ncutoff)
#     energy2=np.zeros(Ncutoff)
#     energy3=np.zeros(Ncutoff)
#     for n in range(0,Ncutoff):
#         energy=Energy(n,B)
#         if (mu-energy[0])/T > 150:
#             energy1[n]=(mu-energy[0])/T-np.log(1+np.exp((mu+2*m*pow(v,2))/T))
#             energy2[n]=np.log(1+np.exp((mu-energy[1])/T))-np.log(1+np.exp((mu+2*m*pow(v,2))/T))
#             energy3[n]=np.log(1+np.exp((mu-energy[2])/T))
#         else:
#             energy1[n]=np.log(1+np.exp((mu-energy[0])/T))-np.log(1+np.exp((mu+2*m*pow(v,2))/T))
#             energy2[n]=np.log(1+np.exp((mu-energy[1])/T))-np.log(1+np.exp((mu+2*m*pow(v,2))/T))
#             energy3[n]=np.log(1+np.exp((mu-energy[2])/T))
        
#     Phi=-T*(np.sum(energy2)+np.sum(energy1)+np.sum(energy3))
#     return Phi*B
# @jit(nopython=True)
def GrandPotential(B,mu,T):
    Ncutoff = int(np.floor(N_0_cutoff/B))
    energy1=np.zeros(Ncutoff)
    energy2=np.zeros(Ncutoff)
    energy3=np.zeros(Ncutoff)
    for n in range(0,Ncutoff):
        energy=Energy(n,B)
        if (mu-energy[0])/T > 150:
            energy1[n]=(mu-energy[0])/T-(mu+2*m*pow(v,2))/T
            energy2[n]=np.log(1+np.exp((mu-energy[1])/T))-(mu+2*m*pow(v,2))/T
            energy3[n]=np.log(1+np.exp((mu-energy[2])/T))
        else:
            energy1[n]=np.log(1+np.exp((mu-energy[0])/T))-(mu+2*m*pow(v,2))/T
            energy2[n]=np.log(1+np.exp((mu-energy[1])/T))-(mu+2*m*pow(v,2))/T
            energy3[n]=np.log(1+np.exp((mu-energy[2])/T))
        
    Phi=-T*(np.sum(energy2)+np.sum(energy1)+np.sum(energy3))
    return Phi*B
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


def GP_LL_contri_zeroT(n,B):
    energy = Energy(n, B)
    contri=0
    if mu > -2*m*v**2:
        if energy[0] < mu:
            contri += energy[0] + 2*m*v**2
        else:
            contri += mu + 2*m*v**2
        if energy[1] < mu:
            contri += energy[1] + 2*m*v**2
        else:
            contri += mu + 2*m*v**2 
    else:
        if energy[0] < mu:
            contri += energy[0] - mu
    if energy[2] < mu:
        contri += energy[2] - mu
    return contri
def GP_LL_contri_finiteT(n,B,T):
    energy=Energy(n,B)
    contri=0
    if (mu-energy[0])/T > 150:
        energy0=(mu-energy[0])/T-(mu+2*m*pow(v,2))/T
    else:
        energy0=np.log(1+np.exp((mu-energy[0])/T))-(mu+2*m*pow(v,2))/T
    if (mu-energy[1])/T > 150:
        energy1=(mu-energy[1])/T-(mu+2*m*pow(v,2))/T
    else:
        energy1=np.log(1+np.exp((mu-energy[1])/T))-(mu+2*m*pow(v,2))/T
    energy2=np.log(1+np.exp((mu-energy[2])/T))
    contri+=-T*(energy1+energy2+energy0)
    return contri

# B_inverse_step = 0.002
# magnetic_field_inverse_range = np.arange(10, 11, B_inverse_step)

# grand_potential_zeroT = np.array(Parallel(n_jobs=-1)(delayed(GrandPotentialZeroTWithPV)(1/b, mu) for b in tqdm(magnetic_field_inverse_range, desc="Calculating Grand Potential")))
# grand_potential_finiteT = np.array(Parallel(n_jobs=-1)(delayed(GrandPotential)(1/b, mu, 0.001) for b in tqdm(magnetic_field_inverse_range, desc="Calculating Grand Potential")))


# num_LL_below_mu = np.array(Parallel(n_jobs=-1)(delayed(LL_Counting)(1/b, mu) for b in tqdm(magnetic_field_inverse_range, desc="Calculating Number of LL below mu")))

# plt.plot(magnetic_field_inverse_range, grand_potential_zeroT)
# plt.xlabel("1/B")
# plt.title(f"total phi,mu={mu},1/B step={B_inverse_step},T=0")
# plt.show()

# plt.plot(magnetic_field_inverse_range, grand_potential_zeroT)
# plt.xlabel("1/B")
# plt.title(f"total phi,mu={mu},1/B step={B_inverse_step},T{0.001}")
# plt.show()

Ncutoff = int(np.floor(N_0_cutoff/B))
energy0=np.zeros(Ncutoff)
energy1=np.zeros(Ncutoff)
energy2=np.zeros(Ncutoff)
Phi_finiteT=0
Phi_finiteT_record=np.zeros(Ncutoff)
for n in range(0,Ncutoff):
    energy=Energy(n,B)
    if (mu-energy[0])/T > 150:
        energy0[n]=(mu-energy[0])/T-(mu+2*m*pow(v,2))/T
    else:
        energy0[n]=np.log(1+np.exp((mu-energy[0])/T))-(mu+2*m*pow(v,2))/T
    if (mu-energy[1])/T > 150:
        energy1[n]=(mu-energy[1])/T-(mu+2*m*pow(v,2))/T
    else:
        energy1[n]=np.log(1+np.exp((mu-energy[1])/T))-(mu+2*m*pow(v,2))/T
    energy2[n]=np.log(1+np.exp((mu-energy[2])/T))
    Phi_finiteT+=-T*(energy0[n]+energy1[n]+energy2[n])
    Phi_finiteT_record[n]=Phi_finiteT
Ncutoff = int(np.floor(N_0_cutoff/B))
Phi_zeroT = 0
Phi_zeroT_record=np.zeros(Ncutoff)
for n in range(Ncutoff):
    energy = Energy(n, B)
    if mu > -2*m*v**2:
        if energy[0] < mu:
            Phi_zeroT += energy[0] + 2*m*v**2
        else:
            Phi_zeroT += mu + 2*m*v**2
        if energy[1] < mu:
            Phi_zeroT += energy[1] + 2*m*v**2
        else:
            Phi_zeroT += mu + 2*m*v**2 
    else:
        if energy[0] < mu:
            Phi_zeroT += energy[0] - mu
    if energy[2] < mu:
        Phi_zeroT += energy[2] - mu
    Phi_zeroT_record[n]=Phi_zeroT

plt.plot(range(0,Ncutoff),Phi_finiteT_record) 
plt.plot(range(0,Ncutoff),Phi_zeroT_record) 
plt.show()
plt.plot(range(0,Ncutoff),Phi_finiteT_record-Phi_zeroT_record) 
plt.show()
contri_finiteT=np.zeros(Ncutoff)
contri_zeroT=np.zeros(Ncutoff)
for n in range(Ncutoff):
    contri_finiteT[n]=GP_LL_contri_finiteT(n,B,T)
    contri_zeroT[n]=GP_LL_contri_zeroT(n,B)
plt.plot(range(0,Ncutoff),contri_finiteT) 
plt.plot(range(0,Ncutoff),contri_zeroT) 
plt.show()
plt.plot(range(0,Ncutoff),contri_finiteT-contri_zeroT) 
plt.show()
