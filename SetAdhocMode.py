import subprocess

def set_adhoc_mode(interface, ip_address):
    """ Set a network interface to ad hoc mode with a specified IP address. """
    try:
        # Set the interface down
        subprocess.run(['sudo', 'ip', 'link', 'set', interface, 'down'], check=True)
        
        # Set the wireless mode to adhoc
        subprocess.run(['sudo', 'iwconfig', interface, 'mode', 'ad-hoc'], check=True)
        
        # Set the IP address
        subprocess.run(['sudo', 'ip', 'addr', 'add', ip_address, 'dev', interface], check=True)
        
        # Set the interface up
        subprocess.run(['sudo', 'ip', 'link', 'set', interface, 'up'], check=True)

        print(f"Interface {interface} is set to ad hoc mode with IP {ip_address}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

# Example usage
# set_adhoc_mode("wlan0", "192.168.1.1/24")
