def function_library(name: str) -> callable:
    def add(x: int, y: int) -> int:
        return x + y

    def subtract(x: int, y: int) -> int:
        return x - y

    def multiply(x: int, y: int) -> int:
        return x * y

    def divide(x: int, y: int) -> float:
        if y == 0:
            raise ValueError("Cannot divide by zero!")
        return x / y

    if name == "add":
        return add
    elif name == "subtract":
        return subtract
    elif name == "multiply":
        return multiply
    elif name == "divide":
        return divide
    else:
        raise ValueError("Invalid function name!")

# Example usage:
add_func = function_library("add")
print(add_func(2, 3))  # Output: 5

subtract_func = function_library("subtract")
print(subtract_func(5, 2))  # Output: 3

multiply_func = function_library("multiply")
print(multiply_func(4, 5))  # Output: 20

divide_func = function_library("divide")
print(divide_func(10, 2))  # Output: 5.0