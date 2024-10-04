import numpy as np
import matplotlib.pyplot as plt

class Domain:
    def __init__(self, points):
        self.points = points
        self.subdomains = []

def rcb(domain, depth, max_depth, processor_id=0, axis=0):
    if depth >= max_depth or len(domain.points) <= 1:
        domain.processor_id = processor_id
        return

    points = np.array(domain.points)
    axis = np.argmax(np.ptp(points, axis=0))  # Axis with the greatest range

    median = np.median(points[:, axis])
    left_points = points[points[:, axis] <= median]
    right_points = points[points[:, axis] > median]

    left_subdomain = Domain(left_points.tolist())
    right_subdomain = Domain(right_points.tolist())
    domain.subdomains = [left_subdomain, right_subdomain]

    rcb(left_subdomain, depth + 1, max_depth, processor_id, axis)
    rcb(right_subdomain, depth + 1, max_depth, processor_id + 1, axis)

def plot_domain(domain, ax, color):
    if hasattr(domain, 'processor_id'):
        points = np.array(domain.points)
        ax.scatter(points[:, 0], points[:, 1], c=color[domain.processor_id % len(color)])
    else:
        for subdomain in domain.subdomains:
            plot_domain(subdomain, ax, color)

# Example usage
np.random.seed(0)
points = np.random.rand(100, 2) * 100
initial_domain = Domain(points.tolist())

# Number of processors
num_processors = 4
max_depth = int(np.ceil(np.log2(num_processors)))

rcb(initial_domain, 0, max_depth)

# Plotting the result
fig, ax = plt.subplots()
colors = ['red', 'blue', 'green', 'purple']
plot_domain(initial_domain, ax, colors)
plt.show()
