import numpy as np
from scipy.signal import convolve2d

class CyberneticPurificationHopfieldNetwork:
    def __init__(self, input_shape, filter_size, num_filters, purity_threshold=0.95):
        self.input_shape = input_shape
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.01
        self.patterns = []
        self.purity_threshold = purity_threshold

    def energy(self, state):
        """Calculate a more complex energy of the current state."""
        convolved_state = self.convolve(state)
        # A more complex energy function could include non-linear interactions
        energy = -0.5 * np.sum(state * convolved_state) + 0.05 * np.sum(np.abs(state - convolved_state)**2)
        return energy

    def convolve(self, state):
        """Apply convolution to the state."""
        convolved = np.zeros_like(state, dtype=np.float64)
        for i in range(self.num_filters):
            convolved += convolve2d(state, self.filters[i], mode='same')
        return convolved

    def store_pattern(self, pattern):
        """Store a 'pure' pattern into the network."""
        self.patterns.append(pattern)

    def purify(self, initial_state, num_iterations=100, initial_learning_rate=0.01):
        """Purify the input state by converging it towards a stored pure state."""
        state = initial_state.astype(np.float64)
        learning_rate = initial_learning_rate

        for iteration in range(num_iterations):
            prev_energy = self.energy(state)
            convolved_state = self.convolve(state)
            state += learning_rate * convolved_state
            state = np.sign(state)

            current_energy = self.energy(state)
            energy_change = np.abs(current_energy - prev_energy)

            if energy_change < 1e-4:
                learning_rate *= 1.05  # Increase learning rate if not much is changing
            else:
                learning_rate *= 0.95  # Otherwise, decrease it to fine-tune

            # Add stochastic component
            if iteration % 10 == 0:
                state += np.random.randn(*state.shape) * 0.1
                state = np.sign(state)

            purity = self.calculate_purity(state)
            if purity >= self.purity_threshold:
                print(f"Converged at iteration {iteration} with purity {purity:.2f}")
                break

        return state

    def calculate_purity(self, state):
        """Calculate how close the current state is to a stored pattern."""
        max_similarity = 0
        for pattern in self.patterns:
            similarity = np.sum(state == pattern) / state.size
            max_similarity = max(max_similarity, similarity)
        return max_similarity

# Example usage
if __name__ == "__main__":
    input_shape = (10, 10)
    filter_size = 3
    num_filters = 5

    # Initialize the Purification Hopfield Network
    chn = CyberneticPurificationHopfieldNetwork(input_shape, filter_size, num_filters)

    # Store a 'pure' pattern representing a clean state
    pure_pattern = np.ones((10, 10))
    chn.store_pattern(pure_pattern)
    print("Stored pure pattern:\n", pure_pattern)

    # Create a noisy or 'impure' state
    impure_state = pure_pattern + np.random.randn(10, 10) * 0.5
    impure_state = np.sign(impure_state)
    print("Impure input state:\n", impure_state)

    # Purify the impure state
    purified_state = chn.purify(impure_state)
    print("Purified state:\n", purified_state)

    # Compare purity before and after
    initial_purity = chn.calculate_purity(impure_state)
    final_purity = chn.calculate_purity(purified_state)
    print(f"Initial purity: {initial_purity:.2f}, Final purity: {final_purity:.2f}")


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Define the Convolutional Hopfield Network class
class ConvolutionalHopfieldNetwork:
    def __init__(self, input_shape, filter_size, num_filters):
        self.input_shape = input_shape
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.01
        self.patterns = []

    def energy(self, state):
        """Calculate the energy of the current state"""
        convolved_state = self.convolve(state)
        energy = -0.5 * np.sum(state * convolved_state)
        return energy

    def convolve(self, state):
        """Apply convolution to the state"""
        convolved = np.zeros_like(state, dtype=np.float64)  # Ensure float64 type
        for i in range(self.num_filters):
            convolved += convolve2d(state, self.filters[i], mode='same')
        return convolved


    def store_pattern(self, pattern):
        """Store a pattern into the network"""
        self.patterns.append(pattern)

    def retrieve_pattern(self, initial_state, num_iterations=10, learning_rate=0.1):
        """Retrieve a pattern given an initial state"""
        state = initial_state.astype(np.float64)  # Ensure state is float64
        for _ in range(num_iterations):
            convolved_state = self.convolve(state)
            state += learning_rate * convolved_state
            state = np.sign(state)
            if self.energy(state) < self.energy(initial_state):
                break
        return state


# Example usage
if __name__ == "__main__":
    input_shape = (28, 28)
    filter_size = 3
    num_filters = 8

    # Initialize the Convolutional Hopfield Network
    chn = ConvolutionalHopfieldNetwork(input_shape, filter_size, num_filters)

    # Create a random pattern to store
    pattern = np.random.choice([-1, 1], size=input_shape)
    chn.store_pattern(pattern)

    # Generate an initial state close to the pattern
    initial_state = pattern + np.random.randn(*input_shape) * 0.2
    initial_state = np.sign(initial_state)

    # Retrieve the stored pattern from the noisy initial state
    retrieved_pattern = chn.retrieve_pattern(initial_state)

    # Display the patterns
    plt.subplot(1, 3, 1)
    plt.title('Original Pattern')
    plt.imshow(pattern, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('Noisy Input')
    plt.imshow(initial_state, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Retrieved Pattern')
    plt.imshow(retrieved_pattern, cmap='gray')

    plt.show()

import numpy as np

# Initialize the Convolutional Hopfield Network
chn = ConvolutionalHopfieldNetwork(input_shape=(10, 10), filter_size=3, num_filters=5)

# 1. Create and store a simple pattern
pattern1 = np.ones((10, 10))
chn.store_pattern(pattern1)
print("Stored pattern 1:\n", pattern1)

# 2. Store a random pattern
pattern2 = np.random.choice([-1, 1], size=(10, 10))
chn.store_pattern(pattern2)
print("Stored pattern 2:\n", pattern2)

# 3. Store a checkerboard pattern
pattern3 = np.indices((10, 10)).sum(axis=0) % 2 * 2 - 1
chn.store_pattern(pattern3)
print("Stored pattern 3 (checkerboard):\n", pattern3)

# 4. Retrieve pattern 1 with slight noise
noisy_pattern1 = pattern1 + np.random.randn(10, 10) * 0.2
noisy_pattern1 = np.sign(noisy_pattern1)
retrieved_pattern1 = chn.retrieve_pattern(noisy_pattern1)
print("Noisy input for pattern 1:\n", noisy_pattern1)
print("Retrieved pattern 1:\n", retrieved_pattern1)

# 5. Retrieve pattern 2 with heavy noise
noisy_pattern2 = pattern2 + np.random.randn(10, 10) * 0.5
noisy_pattern2 = np.sign(noisy_pattern2)
retrieved_pattern2 = chn.retrieve_pattern(noisy_pattern2)
print("Noisy input for pattern 2:\n", noisy_pattern2)
print("Retrieved pattern 2:\n", retrieved_pattern2)

# 6. Energy of the original pattern 1
energy_pattern1 = chn.energy(pattern1)
print("Energy of original pattern 1:", energy_pattern1)

# 7. Energy of the noisy version of pattern 1
energy_noisy_pattern1 = chn.energy(noisy_pattern1)
print("Energy of noisy pattern 1:", energy_noisy_pattern1)

# 8. Compare energy before and after retrieval for pattern 1
energy_retrieved_pattern1 = chn.energy(retrieved_pattern1)
print("Energy of retrieved pattern 1:", energy_retrieved_pattern1)

# 9. Retrieve pattern 3 after introducing noise
noisy_pattern3 = pattern3 + np.random.randn(10, 10) * 0.3
noisy_pattern3 = np.sign(noisy_pattern3)
retrieved_pattern3 = chn.retrieve_pattern(noisy_pattern3)
print("Noisy input for pattern 3:\n", noisy_pattern3)
print("Retrieved pattern 3:\n", retrieved_pattern3)

# 10. Store a pattern with a diagonal line
pattern4 = np.eye(10) * 2 - 1
chn.store_pattern(pattern4)
print("Stored pattern 4 (diagonal line):\n", pattern4)

# 11. Retrieve pattern 4 after small perturbations
perturbed_pattern4 = pattern4 + np.random.randn(10, 10) * 0.1
perturbed_pattern4 = np.sign(perturbed_pattern4)
retrieved_pattern4 = chn.retrieve_pattern(perturbed_pattern4)
print("Perturbed input for pattern 4:\n", perturbed_pattern4)
print("Retrieved pattern 4:\n", retrieved_pattern4)

# 12. Attempt to retrieve a pattern that is not stored
random_input = np.random.choice([-1, 1], size=(10, 10))
retrieved_random = chn.retrieve_pattern(random_input)
print("Random input (not stored):\n", random_input)
print("Retrieved pattern from random input:\n", retrieved_random)

# 13. Store and retrieve a complex pattern
pattern5 = np.zeros((10, 10))
pattern5[2:8, 2:8] = 1
pattern5[4:6, 4:6] = -1
chn.store_pattern(pattern5)
noisy_pattern5 = pattern5 + np.random.randn(10, 10) * 0.2
noisy_pattern5 = np.sign(noisy_pattern5)
retrieved_pattern5 = chn.retrieve_pattern(noisy_pattern5)
print("Stored pattern 5 (complex):\n", pattern5)
print("Noisy input for pattern 5:\n", noisy_pattern5)
print("Retrieved pattern 5:\n", retrieved_pattern5)

# 14. Check the effect of learning rate on retrieval
chn_low_lr = ConvolutionalHopfieldNetwork(input_shape=(10, 10), filter_size=3, num_filters=5)
chn_low_lr.store_pattern(pattern2)
retrieved_low_lr = chn_low_lr.retrieve_pattern(noisy_pattern2, learning_rate=0.01)
print("Retrieved pattern 2 with low learning rate:\n", retrieved_low_lr)

# 15. Visualize energy landscape by printing energy values for different states
for i in range(-3, 4):
    test_pattern = pattern1 + i * 0.1 * np.ones((10, 10))
    test_pattern = np.sign(test_pattern)
    energy_value = chn.energy(test_pattern)
    print(f"Energy of test pattern (perturbation level {i}): {energy_value}")
