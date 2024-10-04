from ping3 import ping, verbose_ping

def measure_network_delay(host):
    """
    Measure the network delay to the specified host.

    Parameters:
    host (str): The host to ping.

    Returns:
    float: The round-trip time in milliseconds.
    """
    delay = ping(host)
    if delay is None:
        print(f"Failed to reach {host}.")
        return None
    else:
        print(f"Network delay to {host}: {delay * 1000:.2f} ms")
        return delay * 1000

if __name__ == "__main__":
    host = 'google.com'  # Change to the host you want to measure delay to
    measure_network_delay(host)
