def temperature_category(temp_celsius: float) -> str:
    if temp_celsius >= 30:
        return "hot"
    elif temp_celsius <= 0:
        return "cold"
    else:
        return "warm"

# Example usage:
temp_celsius = 25.5
print(temperature_category(temp_celsius))  # Output: warm

temp_celsius = 35.5
print(temperature_category(temp_celsius))  # Output: hot

temp_celsius = -5.5
print(temperature_category(temp_celsius))  # Output: cold