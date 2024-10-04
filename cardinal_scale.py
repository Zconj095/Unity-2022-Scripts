def cardinal_scale(cardinal_direction: str) -> str:
    cardinal_directions = {
        "north": "N",
        "northeast": "NE",
        "east": "E",
        "southeast": "SE",
        "south": "S",
        "southwest": "SW",
        "west": "W",
        "northwest": "NW"
    }
    return cardinal_directions.get(cardinal_direction, "Unknown direction")

# Example usage:
print(cardinal_scale("north"))  # Output: "N"
print(cardinal_scale("east"))  # Output: "E"
print(cardinal_scale("unknown"))  # Output: "Unknown direction"