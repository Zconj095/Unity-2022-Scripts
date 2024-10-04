import numpy as np

def memory_subjection(m, i, s, f):
    """
    Calculates the memory subjection based on the given equation.

    Args:
        m: Original memory (numpy array).
        i: Internal subjections (numpy array).
        s: External subjections (numpy array).
        f: Function representing the retrieval process (custom function).

    Returns:
        ms: Memory subjection (numpy array).
    """

    # Calculate the interaction between memory and external influences
    interaction = np.dot(m, s)

    # Combine internal and external influences
    combined_influences = i + interaction

    # Apply the retrieval function to the combined influences
    ms = f(combined_influences)

    return ms

# Example usage
m = np.array([0.5, 0.3, 0.2])  # Original memory
i = np.array([0.1, 0.2, 0.3])  # Internal subjections
s = np.array([0.4, 0.5, 0.6])  # External subjections

# Define a custom retrieval function (e.g., sigmoid)
def retrieval_function(x):
    return 1 / (1 + np.exp(-x))

# Calculate the memory subjection
ms = memory_subjection(m, i, s, retrieval_function)

print("Memory subjection:", ms)
