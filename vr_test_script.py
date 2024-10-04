import openvr
import inputs
import time
import math

# Configuration settings (you can adjust these values)
MAX_ROTATION = math.radians(90)  # Maximum rotation in radians
ROTATION_SMOOTHNESS = 0.75  # Between 0 and 1, higher is smoother
MAX_ROTATION_SMOOTHNESS = 1.0  # Maximum smoothness value
ROTATION_ACCELERATION = 0.01  # Rotation speed increase per frame
MAX_ROTATION_SPEED = 0.05  # Maximum rotation speed per frame
MAX_ROTATION_ACCELERATION = 0.04  # Maximum rotation acceleration per frame

current_rotation_speed = 0  # Current rotation speed
current_rotation = 0  # Current rotation angle

VIRTUAL_ANALOG_TOGGLE_BUTTON = "BTN_TOGGLE"  # Replace with the actual button
use_virtual_analog = False  # Toggled state
virtual_analog_angle = 0  # For simulating movement


def get_headset_pose(vr_system):
    poses = vr_system.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0, 
        openvr.k_unMaxTrackedDeviceCount
    )

    hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]
    if hmd_pose.bPoseIsValid:
        mat = hmd_pose.mDeviceToAbsoluteTracking
        # Position scaling for sensitivity adjustment
        position = (mat[0][3] * 0.1, mat[1][3] * 0.1, mat[2][3] * 0.1)
        # Extract orientation (simplified)
        rotation = math.atan2(mat[0][2], mat[0][0])
        return position, rotation

    return None, None

def process_controller_input():
    global use_virtual_analog
    events = inputs.get_gamepad()
    for event in events:
        if event.ev_type == 'Key' and event.code == VIRTUAL_ANALOG_TOGGLE_BUTTON:
            if event.state == 1:  # Button pressed
                use_virtual_analog = not use_virtual_analog
                print(f"Virtual analog {'enabled' if use_virtual_analog else 'disabled'}.")

        # Process other inputs
        # ...

def get_virtual_analog_input():
    global virtual_analog_angle
    # Simulating a circular movement for the virtual analog
    virtual_analog_x = math.cos(virtual_analog_angle)
    virtual_analog_y = math.sin(virtual_analog_angle)
    virtual_analog_angle += 0.1  # Adjust the speed of the virtual movement here

    return virtual_analog_x, virtual_analog_y

def update_rotation(headset_rotation):
    global current_rotation_speed, current_rotation
    # Calculate rotation delta
    rotation_delta = min(MAX_ROTATION, abs(headset_rotation - current_rotation))
    
    # Apply smoothness
    rotation_delta *= ROTATION_SMOOTHNESS

    # Accelerate rotation speed
    current_rotation_speed += ROTATION_ACCELERATION
    current_rotation_speed = min(current_rotation_speed, MAX_ROTATION_SPEED)

    # Apply rotation speed and acceleration
    rotation_speed = min(rotation_delta, current_rotation_speed)
    rotation_speed = min(rotation_speed, MAX_ROTATION_ACCELERATION)

    # Update current rotation
    if headset_rotation > current_rotation:
        current_rotation += rotation_speed
    elif headset_rotation < current_rotation:
        current_rotation -= rotation_speed

    current_rotation = max(-MAX_ROTATION, min(MAX_ROTATION, current_rotation))  # Clamp rotation

    return current_rotation

def main():
    vr_system = openvr.init(openvr.VRApplication_Scene)
    try:
        while True:
            headset_position, headset_rotation = get_headset_pose(vr_system)
            if headset_position:
                print("Headset Position:", headset_position)

            if headset_rotation:
                smooth_rotation = update_rotation(headset_rotation)
                print("Smooth Headset Rotation:", smooth_rotation)

            headset_position, headset_rotation = get_headset_pose(vr_system)
            if headset_position and headset_rotation:
                print("Headset Position:", headset_position)
                print("Headset Rotation:", headset_rotation)

            process_controller_input()

            if use_virtual_analog:
                virtual_analog_x, virtual_analog_y = get_virtual_analog_input()
                # Here you would process the virtual analog input
                print("Virtual Analog X:", virtual_analog_x, "Y:", virtual_analog_y)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Exiting program...")

    finally:
        openvr.shutdown()

if __name__ == "__main__":
    main()


