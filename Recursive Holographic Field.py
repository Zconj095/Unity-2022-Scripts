import numpy as np
import matplotlib.pyplot as plt

class RecursiveHolographicField:
    def __init__(self, size, depth):
        self.size = size
        self.depth = depth
        self.field = np.zeros((size, size))
    
    def construct_field(self, x0, y0, x1, y1, current_depth):
        if current_depth == self.depth:
            return
        
        # Compute field value for the region
        self.field[x0:x1, y0:y1] = self.compute_field_value(x0, y0, x1, y1)
        
        # Recursively divide the region
        mx = (x0 + x1) // 2
        my = (y0 + y1) // 2
        
        self.construct_field(x0, y0, mx, my, current_depth + 1)
        self.construct_field(mx, y0, x1, my, current_depth + 1)
        self.construct_field(x0, my, mx, y1, current_depth + 1)
        self.construct_field(mx, my, x1, y1, current_depth + 1)
    
    def compute_field_value(self, x0, y0, x1, y1):
        # Simple field value computation, e.g., based on region size or position
        return np.random.rand()
    
    def visualize_field(self):
        plt.imshow(self.field, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.show()

# Example usage
size = 128  # Size of the field (128x128 grid)
depth = 4   # Recursion depth

holographic_field = RecursiveHolographicField(size, depth)
holographic_field.construct_field(0, 0, size, size, 0)
holographic_field.visualize_field()
