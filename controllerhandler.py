import bpy
import bge
import time
from mathutils import Vector
from inputs import get_gamepad

# Function to handle controller input
def handle_controller_input(cont, cube):
    try:
        events = get_gamepad()
        for event in events:
            if event.ev_type == "Absolute":
                if event.code == "ABS_X":  # Left stick horizontal
                    cube.applyMovement((event.state / 32767.0, 0, 0), True)
                elif event.code == "ABS_Y":  # Left stick vertical
                    cube.applyMovement((0, -event.state / 32767.0, 0), True)
            elif event.ev_type == "Key":
                if event.code == "BTN_SOUTH" and event.state == 1:  # 'Y' button pressed
                    cube.applyMovement((0, 0, 0.5), True)
            # Ensure the cube falls back down to the ground
            if cube.worldPosition.z > 0:
                cube.applyMovement((0, 0, -0.01), True)
        time.sleep(0.01)
    except Exception as e:
        print(f"Debug: Controller disconnected. Error: {e}")
        return False
    return True

def main():
    scene = bge.logic.getCurrentScene()
    cube = scene.objects['Cube']
    
    while True:
        if not handle_controller_input(bge.logic.getCurrentController(), cube):
            print("Attempting to reconnect...")
            time.sleep(5)  # Wait before attempting to reconnect

# Run the main function
main()
