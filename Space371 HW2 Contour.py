import numpy as np
import matplotlib.pyplot as plt

# Define the function |E|
def E_magnitude(u, v):
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc_u = np.sin(u) / u
        sinc_v = np.sin(v) / v
    sinc_u[np.abs(u) < 1e-10] = 1.0
    sinc_v[np.abs(v) < 1e-10] = 1.0
    return np.abs(sinc_u * sinc_v)

# Create a grid
u = np.linspace(-15, 15, 500)
v = np.linspace(-15, 15, 500)
U, V = np.meshgrid(u, v)
Z = E_magnitude(U, V)

# Contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(U, V, Z, levels=50, cmap='viridis')
plt.colorbar(contour, label='|E(θ, φ)|')
plt.xlabel('u = 2πkθ')
plt.ylabel('v = 2πkφ')
plt.title('Contour plot |E(θ, φ)|, Square Aperture')
plt.grid(True)
plt.show()