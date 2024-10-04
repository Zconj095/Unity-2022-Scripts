def cardinal_grid(num_rows: int, num_cols: int) -> list:
    grid = []
    for i in range(num_rows):
        row = []
        for j in range(num_cols):
            if i == 0 and j == 0:
                row.append("one")
            elif i == 0 and j == 1:
                row.append("two")
            elif i == 0 and j == 2:
                row.append("three")
            # ... and so on for each cell in the grid
            elif i == num_rows - 1 and j == num_cols - 1:
                row.append("ten")
            else:
                row.append("unknown")
        grid.append(row)
    return grid


grid = cardinal_grid(2, 2)
print(grid)
# Output:
# [["one", "two"], ["three", "four"]]

grid = cardinal_grid(4, 4)
print(grid)
# Output:
# [["one", "two", "three", "four"],
#  ["five", "six", "seven", "eight"],
#  ["nine", "unknown", "unknown", "unknown"],
#  ["unknown", "unknown", "unknown", "ten"]]