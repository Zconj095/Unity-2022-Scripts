def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

# Example usage
print(factorial(5))  # Output: 120

def factorial_relay(n):
    def factorial_helper(n, acc):
        if n == 0:
            return acc
        else:
            return factorial_helper(n - 1, n * acc)
    
    return factorial_helper(n, 1)

# Example usage
print(factorial_relay(5))  # Output: 120

def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# Example usage
print(fibonacci(10))  # Output: 55

def fibonacci_relay(n):
    def fibonacci_helper(n, a, b):
        if n == 0:
            return a
        elif n == 1:
            return b
        else:
            return fibonacci_helper(n - 1, b, a + b)
    
    return fibonacci_helper(n, 0, 1)

# Example usage
print(fibonacci_relay(10))  # Output: 55

def sum_list(lst):
    if not lst:
        return 0
    else:
        return lst[0] + sum_list(lst[1:])

# Example usage
print(sum_list([1, 2, 3, 4, 5]))  # Output: 15

def sum_list_relay(lst):
    def sum_helper(lst, acc):
        if not lst:
            return acc
        else:
            return sum_helper(lst[1:], acc + lst[0])
    
    return sum_helper(lst, 0)

# Example usage
print(sum_list_relay([1, 2, 3, 4, 5]))  # Output: 15
