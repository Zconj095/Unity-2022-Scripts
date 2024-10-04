import openvr
import inputs
import time
import math

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
    events = inputs.get_gamepad()
    for event in events:
        print(event.ev_type, event.code, event.state)

def main():
    # Initialize VR System
    vr_system = openvr.init(openvr.VRApplication_Scene)

    try:
        # Check if the headset is connected
        if vr_system.isTrackedDeviceConnected(openvr.k_unTrackedDeviceIndex_Hmd):
            print("Headset connected.")

            while True:
                headset_position, headset_rotation = get_headset_pose(vr_system)
                if headset_position and headset_rotation:
                    print("Headset Position:", headset_position)
                    print("Headset Rotation:", headset_rotation)

                process_controller_input()

                time.sleep(0.1)  # Small delay to limit the update rate
        else:
            print("Headset not connected.")

    except KeyboardInterrupt:
        print("Exiting program...")

    finally:
        openvr.shutdown()

if __name__ == "__main__":
    main()