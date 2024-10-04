def ordinal_scale(ordinal_value: int) -> str:
    ordinal_values = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth",
        10: "tenth"
    }
    return ordinal_values.get(ordinal_value, "Unknown ordinal value")

# Example usage:
print(ordinal_scale(1))  # Output: "first"
print(ordinal_scale(5))  # Output: "fifth"
print(ordinal_scale(12))  # Output: "Unknown ordinal value"