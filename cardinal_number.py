def cardinal_number(n: int) -> str:
    if n == 1:
        return "one"
    elif n == 2:
        return "two"
    elif n == 3:
        return "three"
    elif n == 4:
        return "four"
    elif n == 5:
        return "five"
    elif n == 6:
        return "six"
    elif n == 7:
        return "seven"
    elif n == 8:
        return "eight"
    elif n == 9:
        return "nine"
    elif n == 10:
        return "ten"
    else:
        raise ValueError("Unsupported cardinal number")

# Example usage:
print(cardinal_number(1))  # Output: one
print(cardinal_number(5))  # Output: five
print(cardinal_number(10))  # Output: ten