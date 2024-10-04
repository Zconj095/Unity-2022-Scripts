def cartesian_grid(x_coords: list, y_coords: list) -> list:
    grid = []
    for x in x_coords:
        row = []
        for y in y_coords:
            row.append((x, y))
        grid.append(row)
    return grid

# Example usage:
x_coords = [1, 2, 3]
y_coords = [4, 5, 6]
grid = cartesian_grid(x_coords, y_coords)
print(grid)
# Output:
# [[(1, 4), (1, 5), (1, 6)],
#  [(2, 4), (2, 5), (2, 6)],
#  [(3, 4), (3, 5), (3, 6)]]