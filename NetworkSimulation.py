import time
import random

class NetworkSimulation:
    def __init__(self, speed, wavelength):
        self.speed = speed  # Speed of transmission in Kbps
        self.wavelength = wavelength  # Wavelength in nanometers (nm)
        self.time_adjustment_factor = 1  # Factor to adjust packet loss based on time interval

    def set_time_adjustment(self, day_factor):
        # Adjust packet loss based on the given day factor
        self.time_adjustment_factor = day_factor

    def transfer_data(self, data_size):
        """ Simulate data transfer with packet loss. """
        packet_size = 10  # Packet size in KB
        total_packets = data_size / packet_size
        packet_loss_rate = 2 / 1000  # 2 seconds per millisecond

        adjusted_packet_loss_rate = packet_loss_rate * (self.speed / self.wavelength) * self.time_adjustment_factor

        successful_packets = total_packets * (1 - adjusted_packet_loss_rate)
        total_time = (data_size * 8) / self.speed  # Convert KB to Kb for speed calculation

        print(f"Transferred {data_size} KB with {successful_packets:.0f} successful packets out of {total_packets:.0f} in {total_time:.2f} seconds")

# Example usage
network = NetworkSimulation(speed=15, wavelength=500)  # 15 Kbps speed, 500 nm wavelength
network.set_time_adjustment(day_factor=1.2)  # Adjust for a specific day factor

# Simulating data transfer
network.transfer_data(1000)  # Transfer 1000 KB of data
