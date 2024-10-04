import time
import bpy
from inputs import get_gamepad

# Initialize the controller state
controller_active = True

# Function to keep the controller active
def keep_controller_active():
    global controller_active
    while controller_active:
        events = get_gamepad()
        for event in events:
            print(event.ev_type, event.code, event.state)
        time.sleep(1)

# Start the function to keep the controller active
keep_controller_active()
