import time
def cube_rotation_independent(duration, cube_size=10):
    elapsed_time = 0
    rotation = 0  # Initialize rotation at 0 degrees
    max_rotation = 360  # Maximum rotation in degrees
    percentage_completion = 0

    while rotation < max_rotation:
        # Calculate the current dimension of the cube based on the elapsed time
        current_dimension = (elapsed_time / duration) * cube_size if elapsed_time <= duration else cube_size
        percentage_completion = (elapsed_time / duration) * 100 if elapsed_time <= duration else 100
        print(f"Elapsed Time: {elapsed_time} seconds, Rotation: {rotation} degrees, Cube Dimension: {current_dimension} x {current_dimension} x {current_dimension}, Completion: {percentage_completion}%")

        time.sleep(1)  # Wait for one second
        elapsed_time += 1
        rotation += 1  # Increment rotation by 1 degree

    print("Cube Rotation Completed. 365 degrees reached.")

# Example usage: Set a timer duration for 60 seconds for the cube size growth
cube_rotation_independent(60)
