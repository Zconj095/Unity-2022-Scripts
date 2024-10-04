import bpy

def init_controller():
    # Initialize the joystick, assuming a single joystick for simplicity
    try:
        joystick = bpy.context.window_manager.gamepad_devices[0]
        print(f"Joystick {joystick.name} initialized.")
        return joystick
    except IndexError:
        print("No joystick detected.")
        return None

def read_controller(joystick):
    # Read the joystick input
    axis_data = joystick.axis_values
    button_data = joystick.button_values
    
    return axis_data, button_data

def main():
    scene = bpy.context.scene
    obj = scene.objects["YourObjectName"]  # Replace with your object name
    
    if "joystick" not in obj:
        obj["joystick"] = init_controller()
    
    joystick = obj["joystick"]
    if joystick:
        axis_data, button_data = read_controller(joystick)
        
        # Example: Apply axis data to move an object
        # Axis 0 is usually the left stick horizontal movement
        # Axis 1 is usually the left stick vertical movement
        if len(axis_data) >= 2:
            move_x = axis_data[0]
            move_y = axis_data[1]
            
            # Adjust these values as needed to control your object's movement
            obj.location.x += move_x * 0.1
            obj.location.y += move_y * 0.1
            
        # Example: Print button data
        if button_data:
            print(button_data)

# Ensure this script runs continuously in the game engine
bpy.app.handlers.frame_change_post.append(main)
