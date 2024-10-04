# main.py

from get_aura_color import *
from get_personality_traits import *
from display_traits import *
from is_valid_color import *

if __name__ == "__main__":
    aura_color = get_aura_color()
    if is_valid_color(aura_color):
        personality_traits = get_personality_traits(aura_color)
        display_traits(aura_color, personality_traits)
    else:
        print("Entered an invalid aura color. Please try again with a known color.")
