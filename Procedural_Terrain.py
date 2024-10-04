import numpy as np
import matplotlib.pyplot as plt

def generate_terrain(width, height, scale, octaves, seed=None):
    # Initialize the noise array
    noise = np.zeros((width, height))

    # Generate noise at different scales
    for octave in range(octaves):
        frequency = 1.0 / (2 ** octave)

        if seed is not None:
            rng = np.random.RandomState(seed + octave)
        else:
            rng = np.random.RandomState()

        for i in range(width):
            for j in range(height):
                x = i * frequency
                y = j * frequency
                noise[i, j] += rng.rand()

    # Normalize the noise
    noise -= np.min(noise)
    noise /= (np.max(noise) - np.min(noise))

    # Scale the noise
    noise *= scale

    return noise

def smooth(noise, radius):
    # Initialize a smoothed copy of the noise
    smoothed_noise = np.copy(noise)

    # Apply a smoothing kernel
    for i in range(noise.shape[0]):
        for j in range(noise.shape[1]):
            total = 0
            count = 0

            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    if (0 <= i + di < noise.shape[0]) and (0 <= j + dj < noise.shape[1]):
                        total += noise[i + di, j + dj]
                        count += 1

            smoothed_noise[i, j] = total / count

    return smoothed_noise

def adjust_height_range(terrain, min_height, max_height):
    terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain))
    terrain *= (max_height - min_height)
    terrain += min_height

    return terrain

def erode_terrain(terrain, erosion_rate):
    for i in range(terrain.shape[0]):
        for j in range(terrain.shape[1]):
            if np.random.rand() < erosion_rate:
                terrain[i, j] -= 0.01

    return terrain

def create_water_bodies(terrain, water_level):
    water_cells = terrain < water_level
    water_body = np.zeros_like(terrain)
    water_body[water_cells] = 1

    return water_body

def add_vegetation(terrain, vegetation_density, vegetation_type):
    vegetation = np.zeros_like(terrain)
    for i in range(terrain.shape[0]):
        for j in range(terrain.shape[1]):
            if np.random.rand() < vegetation_density:
                if vegetation_type == "trees":
                    if terrain[i, j] > water_level:
                        vegetation[i, j] = 1
                elif vegetation_type == "grass":
                    vegetation[i, j] = 1

    return vegetation

def combine_layers(terrain, water_body, vegetation):
    combined_terrain = terrain * (1 - water_body) + water_body * 0.5 + vegetation * 0.2

    return combined_terrain

# Define parameters
width = 200
height = 200
scale = 100
octaves = 4
seed = 1234
water_level = 50
erosion_rate = 0.01
vegetation_density = 0.2
vegetation_type = "trees"

# Generate terrain layers
terrain = generate_terrain(width, height, scale, octaves, seed)
terrain = smooth(terrain, 3)
terrain = adjust_height_range(terrain, 0, 100)
terrain = erode_terrain(terrain, erosion_rate)

water_body = create_water_bodies(terrain, water_level)
vegetation = add_vegetation(terrain, vegetation_density, vegetation_type)

# Combine terrain layers
combined_terrain = combine_layers(terrain, water_body, vegetation)

# Plot the terrain
plt.imshow(combined_terrain)
plt.colorbar()
plt.show()
