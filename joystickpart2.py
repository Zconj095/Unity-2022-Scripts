import bpy

def init_joysticks():
    joysticks = bpy.context.window_manager.gamepad_devices
    
    if len(joysticks) == 0:
        print("No joysticks found.")
        return None, None
    
    if len(joysticks) == 1:
        print(f"One joystick found: {joysticks[0].name}")
        return joysticks[0], None
    
    print(f"Two joysticks found: {joysticks[0].name} and {joysticks[1].name}")
    return joysticks[0], joysticks[1]

def read_joystick(joystick):
    # Read axes and buttons
    axes = joystick.axis_values
    
    # Return left stick (axes 0 and 1) and right stick (axes 2 and 3)
    left_stick = (axes[0], axes[1])
    right_stick = (axes[2], axes[3]) if len(axes) > 3 else (0.0, 0.0)
    
    return left_stick, right_stick

def main():
    scene = bpy.context.scene
    obj = scene.objects["YourObjectName"]  # Replace with your object name
    
    if "joystick1" not in obj:
        obj["joystick1"], obj["joystick2"] = init_joysticks()
    
    joystick1 = obj["joystick1"]
    joystick2 = obj["joystick2"]
    
    if joystick1:
        left_stick1, right_stick1 = read_joystick(joystick1)
        # Apply movement based on left and right stick of joystick1
        move_x1, move_y1 = left_stick1
        look_x1, look_y1 = right_stick1
        
        # Adjust these values as needed to control your object's movement
        obj.location.x += move_x1 * 0.1
        obj.location.y += move_y1 * 0.1
        # You can apply look/rotation if needed, e.g.:
        # obj.rotation_euler.x += look_y1 * 0.05
        # obj.rotation_euler.y += look_x1 * 0.05
        
    if joystick2:
        left_stick2, right_stick2 = read_joystick(joystick2)
        # Apply movement based on left and right stick of joystick2
        move_x2, move_y2 = left_stick2
        look_x2, look_y2 = right_stick2
        
        # Adjust these values as needed to control your object's movement
        obj.location.x += move_x2 * 0.1
        obj.location.y += move_y2 * 0.1
        # You can apply look/rotation if needed, e.g.:
        # obj.rotation_euler.x += look_y2 * 0.05
        # obj.rotation_euler.y += look_x2 * 0.05

# Ensure this script runs continuously in the game engine
bpy.app.handlers.frame_change_post.append(main)
