import numpy as np
import matplotlib.pyplot as plt

print("Libraries imported good")

# Constants
h = 6.626e-34    # J·s
c = 2.998e10     # cm/s
kB = 1.381e-23   # J/K

# Parameters
nu0 = 667        # cm⁻¹
gamma = 5        # cm⁻¹
Ts = 200         # K
Ta = 300         # K
tau0_list = [0.1, 1, 10]

# Wavenumber range
nu = np.linspace(640, 700, 1000)  # cm⁻¹

# Planck function in wavenumber space
def planck(nu, T):
    numerator = 2 * h * c**2 * nu**3
    exponent = h * c * nu / (kB * T)
    denominator = np.exp(exponent) - 1
    return numerator / denominator

# Compute optical depth (Lorentzian)
def tau_nu(nu, tau0, nu0, gamma):
    return tau0 * ( (gamma/2)**2 ) / ( (nu - nu0)**2 + (gamma/2)**2 )

# Compute absorptivity
def alpha_nu(tau_nu):
    return 1 - np.exp(-tau_nu)

# Compute TOA radiance
def I_nu(nu, tau0, nu0, gamma, Ts, Ta):
    tau = tau_nu(nu, tau0, nu0, gamma)
    alpha = alpha_nu(tau)
    B_s = planck(nu, Ts)
    B_a = planck(nu, Ta)
    return (1 - alpha) * B_s + alpha * B_a

# Plot
plt.figure(figsize=(10,6))
for tau0 in tau0_list:
    I = I_nu(nu, tau0, nu0, gamma, Ts, Ta)
    plt.plot(nu, I, label=f'τ₀ = {tau0}')

plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Radiance (W/m²/sr/cm⁻¹)')
plt.title('TOA Radiance Spectrum for Different Optical Depths')
plt.legend()
plt.grid(True)
plt.show()