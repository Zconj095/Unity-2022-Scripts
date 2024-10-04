# personality_traits_mapper.py

def get_personality_traits(aura_color):
    traits_map = {
        "red": "Passionate, energetic, and competitive",
        "orange": "Creative, adventurous, and confident",
        # Add other colors as needed
    }
    return traits_map.get(aura_color.lower(), "Unknown aura color. Unable to determine personality traits.")
