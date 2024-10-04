import random
import time

class NetworkDevice:
    def __init__(self, name, connection_type, position, range, is_edge_device=False):
        self.name = name
        self.connection_type = connection_type
        self.position = position
        self.range = range
        self.is_edge_device = is_edge_device
        self.connected_devices = []
        self.sync_timer = 0  # Timer for synchronization

    def connect_device(self, other_device):
        """ Connect with another device if within range. """
        if self.is_in_range(other_device):
            self.connected_devices.append(other_device.name)
            print(f"{self.name} is now connected to {other_device.name}.")
        else:
            print(f"Connection failed: {other_device.name} is out of range.")

    def is_in_range(self, other_device):
        if self.connection_type == 'Wireless':
            distance = ((self.position[0] - other_device.position[0]) ** 2 + 
                        (self.position[1] - other_device.position[1]) ** 2) ** 0.5
            return distance <= self.range
        else:
            return True  # Wired connections are not limited by range

    def process_data(self, data):
        """ Process data if the device is an edge computing device. """
        if self.is_edge_device:
            processed_data = f"Processed {data} by {self.name}"
            print(processed_data)
            return processed_data
        else:
            print(f"{self.name} is not an edge computing device.")
            return None

    def transfer_data(self, other_device, data):
        """ Transfer data to another connected device. """
        if other_device.name in self.connected_devices:
            self.sync_transfer(other_device)
            time.sleep(self.sync_timer / 1000)  # Sync timer in milliseconds
            print(f"Transferring data from {self.name} to {other_device.name}...")
            processed_data = other_device.process_data(data)
            return processed_data
        else:
            print(f"Transfer failed: {other_device.name} is not connected.")
            return None

    def sync_transfer(self, other_device):
        """ Synchronize transfer to optimize timing based on delays and signals. """
        if self.connection_type == 'Ethernet':
            signal_quality = self.calculate_signal_quality(other_device)
            self.sync_timer = 100 / signal_quality  # Example synchronization logic
            print(f"Synchronization timer set to {self.sync_timer}ms for {self.name}")
        else:
            print("Synchronization is more effective in wired connections.")

    def calculate_signal_quality(self, other_device):
        """ Calculate signal quality (simplified for demonstration). """
        return random.uniform(50, 100)  # Simulated signal quality

# Example usage
device_a = NetworkDevice("Device A", "Ethernet", (0, 0), 50, True)
device_b = NetworkDevice("Device B", "Wireless", (30, 40), 50, False)

# Connect devices and transfer data
device_a.connect_device(device_b)
device_a.transfer_data(device_b, "Sample Data")
