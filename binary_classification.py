def classify(x: float) -> str:
    if x >= 0.5:
        return "Positive"
    else:
        return "Negative"

# Example usage:
print(classify(0.7))  # Output: Positive
print(classify(0.3))  # Output: Negative
print(classify(0.49))  # Output: Negative
print(classify(0.51))  # Output: Positive