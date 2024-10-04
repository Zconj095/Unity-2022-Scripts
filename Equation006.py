import numpy as np

def impure_memory(M, D, G, AS, MS, CR):
    # Model memory transformation based on desires and biases
    desire_weights = D / np.sum(D)  # Normalize desire weights
    distortion = np.dot(desire_weights, np.random.randn(len(M)))  # Introduce distortions based on desires
    biased_memory = M + distortion + AS * np.random.rand() + MS  # Apply biases and randomness

    # Calculate destructive potential based on dominant desires
    destructive_score = np.max(D) - G

    # Combine factors into overall impurity score
    impurity = np.mean(biased_memory) + destructive_score * CR

    return impurity

# Example usage
M = np.array([0.7, 0.8, 0.5])  # Memory components (example)
D = np.array([0.3, 0.5, 0.2])  # Desires (example)
G = 0.1  # Goodwill/Faith (example)
AS = 0.2  # Automatic Subjection (example)
MS = 0.1  # Manual Subjection (example)
CR = 1.2  # Chemical Response factor (example)

impure_score = impure_memory(M, D, G, AS, MS, CR)

print("Impure memory score:", impure_score)
