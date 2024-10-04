class AuraPersonalityReader:
    def __init__(self):
        # Mapping of aura colors to personality traits (fictional for demonstration)
        self.aura_traits_map = {
            "red": "Passionate, energetic, and competitive",
            "orange": "Creative, adventurous, and confident",
            "yellow": "Optimistic, cheerful, and intellectual",
            "green": "Balanced, natural, and stable",
            "blue": "Calm, trustworthy, and communicative",
            "indigo": "Intuitive, curious, and reflective",
            "violet": "Imaginative, visionary, and sensitive",
        }

    def read_aura(self, aura_color):
        # Return the personality traits associated with the aura color
        return self.aura_traits_map.get(aura_color.lower(), "Unknown aura color. Unable to determine personality traits.")

    def display_traits(self, aura_color):
        # Display the personality traits based on the aura color
        personality_traits = self.read_aura(aura_color)
        print(f"Aura Color: {aura_color.capitalize()}\nPersonality Traits: {personality_traits}")

if __name__ == "__main__":
    # Example usage
    aura_reader = AuraPersonalityReader()
    aura_color = input("Enter your aura color to find out your personality traits: ")
    aura_reader.display_traits(aura_color)
