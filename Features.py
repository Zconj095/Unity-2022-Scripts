import numpy as np
from impressive import * 
def generate_terrain_features(terrain, feature_type, feature_density, feature_size):
    features = np.zeros_like(terrain)

    if feature_type == "hills":
        for i in range(terrain.shape[0]):
            for j in range(terrain.shape[1]):
                if np.random.rand() < feature_density:
                    # Generate a hill with random size and position
                    hill_size = np.random.randint(feature_size // 2, feature_size)
                    hill_center_x = np.random.randint(hill_size, terrain.shape[0] - hill_size)
                    hill_center_y = np.random.randint(hill_size, terrain.shape[1] - hill_size)

                    # Create a Gaussian hill function
                    hill_function = np.exp(-((i - hill_center_x)**2 + (j - hill_center_y)**2) / (2 * hill_size**2))

                    # Add the hill function to the features array
                    features[i, j] = hill_function

    elif feature_type == "caves":
        for i in range(terrain.shape[0]):
            for j in range(terrain.shape[1]):
                if np.random.rand() < feature_density:
                    # Generate a cave with random size and position
                    cave_size = np.random.randint(feature_size // 2, feature_size)
                    cave_center_x = np.random.randint(cave_size, terrain.shape[0] - cave_size)
                    cave_center_y = np.random.randint(cave_size, terrain.shape[1] - cave_size)

                    # Create a Gaussian cave function
                    cave_function = 1 - np.exp(-((i - cave_center_x)**2 + (j - cave_center_y)**2) / (2 * cave_size**2))

                    # Subtract the cave function from the terrain to create a sunken area
                    terrain[i, j] -= cave_function

                    # Add the cave function to the features array
                    features[i, j] = cave_function

    return features

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

def create_mountain_range(terrain, mountain_width, mountain_height, mountain_location):
    # Define mountain range parameters
    mountain_base = mountain_location[0]
    mountain_peak = mountain_location[1]

    for i in range(mountain_width):
        for j in range(terrain.shape[1]):
            distance_from_base = abs(mountain_base - i)
            distance_from_peak = abs(mountain_peak - i)

            # Create a Gaussian mountain function
            mountain_function = np.exp(-(distance_from_base**2 + distance_from_peak**2) / (2 * mountain_height**2))

            # Add the mountain function to the terrain
            terrain[i, j] += mountain_function

    return terrain


def create_tundra_layer(terrain):
    # Define tundra elevation range
    tundra_elevation_min = 0.2
    tundra_elevation_max = 0.4

    # Create a tundra mask based on elevation
    tundra_mask = (terrain >= tundra_elevation_min) & (terrain <= tundra_elevation_max)

    # Create a tundra layer
    tundra_layer = np.zeros_like(terrain)
    tundra_layer[tundra_mask] = 1

    return tundra_layer

# Plot the terrain
plt.imshow(combined_terrain)
plt.colorbar()
plt.show()


