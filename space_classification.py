def space_classification(space_type: str) -> str:
    space_types = {
        "object": "object space",
        "local": "local space",
        "level": "level space",
        "global": "global space",
        "null": "undefined space"
    }
    return space_types.get(space_type, "unknown space type")

# Example usage:
print(space_classification("object"))  # Output: "object space"
print(space_classification("local"))  # Output: "local space"
print(space_classification("level"))  # Output: "level space"
print(space_classification("global"))  # Output: "global space"
print(space_classification("unknown"))  # Output: "unknown space type"