import openvr
import inputs
import time
import math
import numpy as np
from pyquaternion import Quaternion
# Existing configuration settings
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

def convert_to_quaternion(yaw):
    # Create a Quaternion object representing rotation around the Y-axis (vertical axis)
    return Quaternion(axis=[0, 1, 0], angle=yaw)




class UserInteraction:
    def __init__(self):
        self.head_position = (0, 0, 0)  # XYZ coordinates for head position
        self.head_orientation = (0, 0, 0, 1)  # Quaternion for head orientation
        self.hand_gestures = {'left': None, 'right': None}  # Track hand gestures for left and right hands

    def update_head_position(self, new_position):
        # Trilinear interpolation can be applied here for smooth position transition
        self.head_position = new_position

    def update_head_orientation(self, new_orientation):
        # SLERP interpolation for smooth orientation transition
        self.head_orientation = self.slerp(self.head_orientation, new_orientation, ROTATION_SMOOTHNESS)

    def slerp(self, start_orientation, end_orientation, t):
        # Ensure both orientations are Quaternions
        start_orientation = Quaternion(start_orientation)
        end_orientation = Quaternion(end_orientation)

        # Manually compute the dot product (cos_theta)
        cos_theta = sum(a * b for a, b in zip(start_orientation.elements, end_orientation.elements))

        # Ensure cos_theta is a float for np.arccos
        theta = np.arccos(max(min(cos_theta, 1), -1))  # Clamp cos_theta to the range [-1, 1] to avoid numerical errors
        sin_theta = np.sin(theta)

        if sin_theta > 0.001:
            interp = lambda a, b, t: (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / sin_theta
            result = interp(start_orientation, end_orientation, t)
            return result.elements  # Return the quaternion as a tuple
        else:
            return start_orientation.elements

# Standalone functions
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
    current_rotation_speed = 0  # Current rotation speed
    current_rotation = 0  # Current rotation angle

    # Update current rotation
    if headset_rotation > current_rotation:
        current_rotation += current_rotation_speed
    elif headset_rotation < current_rotation:
        current_rotation -= current_rotation_speed

    current_rotation = max(-MAX_ROTATION, min(MAX_ROTATION, current_rotation))  # Clamp rotation

    return current_rotation


def main():
    vr_system = openvr.init(openvr.VRApplication_Scene)
    user_interaction = UserInteraction()

    try:
        while True:
            headset_position, headset_yaw = get_headset_pose(vr_system)  # Assuming headset_yaw is a single float value
            if headset_position:
                user_interaction.update_head_position(headset_position)

            if headset_yaw is not None:  # Check if headset_yaw is not None
                quaternion_rotation = convert_to_quaternion(headset_yaw)
                user_interaction.update_head_orientation(quaternion_rotation)
            

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