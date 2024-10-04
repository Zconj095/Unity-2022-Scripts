import bpy
from experience_curve_modifier import ExperienceCurveModifier

# Create an instance of the ExperienceCurveModifier class
modifier = ExperienceCurveModifier(curve_type='logarithmic')

# Get the current character
character = bpy.context.scene.character

# Define the event handler for XP gain
def on_xp_gain(new_xp):
    # Update the XP using the experience curve modifier
    modifier.update_xp(character, new_xp)

# Register the event handler
bpy.app.handlers.game_engine.register_event_handler('XP_GAIN', on_xp_gain)