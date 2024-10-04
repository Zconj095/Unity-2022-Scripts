from openvr import *
import time

def main():
    # Initialize OpenVR
    openvr.init(openvr.VRApplication_Scene)

    # Get the VR system object
    vr_system = openvr.VRSystem()

    # Check if the headset is connected
    if vr_system.isTrackedDeviceConnected(openvr.k_unTrackedDeviceIndex_Hmd):
        print("Headset connected.")

        # Get and print some basic information about the headset
        for device in range(openvr.k_unMaxTrackedDeviceCount):
            if vr_system.isTrackedDeviceConnected(device):
                device_class = vr_system.getTrackedDeviceClass(device)
                print(f"Device {device}: Class {device_class}")

                if device_class == openvr.TrackedDeviceClass_HMD:
                    print(" - This is the Headset")
                    # Add more functionality here if needed

    else:
        print("Headset not connected.")

    # Cleanup OpenVR
    openvr.shutdown()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure OpenVR is properly shutdown
        openvr.shutdown()

from inputs import get_gamepad

def main():
    while True:
        events = get_gamepad()
        for event in events:
            print(event.ev_type, event.code, event.state)

if __name__ == "__main__":
    main()
