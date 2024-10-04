import numpy as np

# Define constants  
c = 3e8 # speed of light (m/s)  
vs = 340 # speed of sound (m/s)
kB = 1.38e-23 # Boltzmann constant
ħ = 1.05e-34 # Reduced Planck's constant
T = 300 # Temperature (K) 

# Characteristic wavelengths  
λs = vs / 5000 # 500 Hz sound wave 
λγ = c / (4e14) # Near visible light

# Energies
Es = kB * T # Average phonon energy at T
Eγ = ħ * c / λγ # Photon energy  

# Subatomic size scale 
r0 = 1e-10 # 10 pm 

# Calculate key ratios
λ_ratio = λγ/λs # Wavelength ratio  
E_ratio = Eγ/Es # Energy ratio
size_ratio = λs/r0 # Phonon wavelength to subatomic scale

print(f'Wavelength Ratio: {λ_ratio:.5g}')
print(f'Energy Ratio: {E_ratio:.5g}') 
print(f'Phonon Size Ratio: {size_ratio:.2f}')

def forces(positions, velocities):
    # Placeholder function to calculate forces
    # This is just a dummy example and should be replaced with actual physics
    num_atoms = positions.shape[0]
    force_matrix = np.zeros((num_atoms, 3))  # Initialize a matrix to store forces
    
    # Example: Simple repulsive force from the origin
    for i in range(num_atoms):
        direction_to_origin = origin - positions[i]
        distance_to_origin = np.linalg.norm(direction_to_origin)
        if distance_to_origin > 0:  # Avoid division by zero
            force_magnitude = 1 / distance_to_origin**2  # Inverse square law as an example
            force_direction = direction_to_origin / distance_to_origin
            force_matrix[i] = force_magnitude * force_direction
    
    return force_matrix