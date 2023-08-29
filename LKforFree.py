import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit

mu=2
Ncutoff = 10000
def Energy(n, B):
    return B*(n+0.5)
def GrandPotential(B, mu, T):
    phi = 0
    n_values = np.arange(Ncutoff)
    energies = Energy(n_values, B)
    phi += T*np.sum(np.log(1 + np.exp((mu - energies) / T)))
    return phi * B
def GrandPotentialZeroT(B, mu):
    phi = 0
    n_values = np.arange(Ncutoff)
    energies = Energy(n_values, B)
    for i in n_values:
        if energies[i] < mu:
            phi += energies[i]-mu
    return phi * B
# Define the function for polynomial fitting
def negative_power_polynomial_func(x, *coefficients):
    y = np.zeros_like(x, dtype=np.float64)
    for i, c in enumerate(coefficients):
        y += c * np.power(x, -i, dtype=np.float64)
    return y
magnetic_field_inverse_range=np.arange(1,5, 0.01)

T_range = np.linspace(0.003, 0.2, 100)
M = np.zeros((len(magnetic_field_inverse_range), len(T_range)))
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
fixed_magnetic_field_inverse_index = 20  # Index of the fixed magnetic field value
fixed_magnetic_field_inverse = magnetic_field_inverse_range[fixed_magnetic_field_inverse_index]

plt.plot(T_range, np.abs(M[fixed_magnetic_field_inverse_index]))
plt.xlabel('Temperature (T)')
plt.ylabel('Oscillatory M')
plt.title(f'Magnetic Field Inverse= {fixed_magnetic_field_inverse}')
plt.show()
