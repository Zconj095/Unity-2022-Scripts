import numpy as np

def split_points(points, dimension):
    sorted_points = points[points[:, dimension].argsort()]
    middle = len(sorted_points) // 2
    return sorted_points[:middle], sorted_points[middle:]

def median_of_dimension(points, dimension):
    sorted_points = points[points[:, dimension].argsort()]
    middle = len(sorted_points) // 2
    return sorted_points[middle]

def recursive_median_point(points):
    if len(points) == 1:
        return points[0]
    
    num_dimensions = points.shape[1]
    medians = []
    
    for dimension in range(num_dimensions):
        left_points, right_points = split_points(points, dimension)
        left_median = recursive_median_point(left_points)
        right_median = recursive_median_point(right_points)
        medians.append(left_median if len(left_points) >= len(right_points) else right_median)
    
    return np.median(medians, axis=0)

# Example usage
points = np.array([
    [2, 3],
    [5, 4],
    [9, 6],
    [4, 7],
    [8, 1],
    [7, 2]
])

median_point = recursive_median_point(points)
print("Median Point:", median_point)
