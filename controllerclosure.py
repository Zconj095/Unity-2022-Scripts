import time
import bpy
from inputs import get_gamepad, devices

# Function to check force feedback availability
def check_force_feedback():
    controllers = devices.gamepads
    for controller in controllers:
        if 'Microsoft X-Box Series X' in controller.name:
            if not controller.has_ff_rumble():
                print(f"Game Controller ({controller.name}) with index {controller.index} has not force feedback (vibration) available")
            return controller
    return None

# Function to handle controller input
def handle_controller_input(controller):
    while True:
        try:
            events = get_gamepad()
            for event in events:
                if event.ev_type == "Absolute":
                    print(event.ev_type, event.code, event.state)
            time.sleep(0.1)
        except Exception as e:
            print(f"Error: {e}. Attempting to reconnect...")
            controller = check_force_feedback()
            if not controller:
                print("Reconnection failed. Exiting.")
                break

# Main function
def main():
    controller = check_force_feedback()
    if controller:
        print(f"Using controller: {controller.name}")
        handle_controller_input(controller)
    else:
        print("No compatible controller found.")

# Run the main function
main()
