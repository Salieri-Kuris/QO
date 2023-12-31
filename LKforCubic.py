import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit

v = 1
m = 0.5
mu1 = -50
W = 5
mu = -0.6
N_0_cutoff = 10000
def Energy(n, B):
    Coefficient = [1, -(mu1 + B * (n + 1/2) / m - 2 * m * pow(v, 2)),
                    -(pow(W, 2) + 2 * B * pow(v, 2) * (2 * n + 3/2) + 2 * m * mu1 * pow(v, 2)),
                    -4 * m * B * pow(v, 4) * (n + 1)]
    sortedroots=np.sort(np.roots(Coefficient))
    return sortedroots

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

# Define the function for polynomial fitting
def negative_power_polynomial_func(x, *coefficients):
    y = np.zeros_like(x, dtype=np.float64)
    for i, c in enumerate(coefficients):
        y += c * np.power(x, -i, dtype=np.float64)
    return y

# Calculate oscillatory grand potential for a range of magnetic fields and temperatures
magnetic_field_inverse_range = np.linspace(1, 1.5, 200)
T_range = np.linspace(0.01,0.6 , 20)
M=np.zeros((len(magnetic_field_inverse_range), len(T_range)))
oscillatory_grand_potential = np.zeros((len(magnetic_field_inverse_range), len(T_range)))

for i, T in enumerate(tqdm(T_range, desc="Calculating Grand Potential")):
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
    
    oscillatory_grand_potential[:,i] = oscillatory_part
    
    M[:,i] = np.gradient(oscillatory_part, magnetic_field_inverse_range)*magnetic_field_inverse_range**2
# Plot the oscillatory grand potential for a fixed magnetic field as a function of T
fixed_magnetic_field_inverse_index = 5  # Index of the fixed magnetic field value
fixed_magnetic_field_inverse = magnetic_field_inverse_range[fixed_magnetic_field_inverse_index]

plt.plot(T_range, np.abs(M[fixed_magnetic_field_inverse_index]))
plt.xlabel('Temperature (T)')
plt.ylabel('Oscillatory M')
plt.title(f'Magnetic Field Inverse= {fixed_magnetic_field_inverse}')
plt.show()
