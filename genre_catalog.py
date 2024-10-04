def genre_catalog(genre: str) -> dict:
    if genre == "fantasy":
        return {"classes": ["ghibli fantasy", "romantic fantasy", "adventure fantasy"],
                "sub-genres": ["relgious", "spiritual", "magical"]}
    elif genre == "fictional":
        return {"classes": ["fictional romance", "science fiction", "fictional fantasy"],
                "sub-genres": ["adventure", "action", "mystical"]}
    else:
        raise ValueError("Unsupported genre")

# Example usage:
catalog = genre_catalog("rock")
print(catalog)  # Output: {"artists": [...], "songs": [...]}

catalog = genre_catalog("pop")
print(catalog)  # Output: {"artists": [...], "songs": [...]}

try:
    genre_catalog("Game Genres")
except ValueError:
    print("Error: Unsupported genre")