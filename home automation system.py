class Device:
    def __init__(self, name):
        self.name = name

class Light(Device):
    def __init__(self, name):
        super().__init__(name)
        self.is_on = False
        self.brightness = 0

    def turn_on(self):
        self.is_on = True
        print(f"{self.name} is now ON.")

    def turn_off(self):
        self.is_on = False
        print(f"{self.name} is now OFF.")

    def set_brightness(self, level):
        self.brightness = level
        print(f"{self.name} brightness is now set to {self.brightness}.")

class Thermostat(Device):
    def __init__(self, name):
        super().__init__(name)
        self.temperature = 20

    def set_temperature(self, temperature):
        self.temperature = temperature
        print(f"{self.name} temperature is now set to {self.temperature}°C.")

class HomeAutomationSystem:
    def __init__(self):
        self.devices = {}

    def add_device(self, device):
        self.devices[device.name.lower()] = device

    def process_command(self, command):
        parts = command.split()
        self._process(parts)

    def _process(self, parts):
        if not parts:
            print("Incomplete command.")
            return

        action = parts[0]
        target = parts[1] if len(parts) > 1 else None

        if target:
            device = self.devices.get(target.lower())
            if not device:
                print(f"No device named '{target}' found.")
                return
        else:
            print("Please specify a target device.")
            return

        if isinstance(device, Light):
            if action == "turn":
                sub_action = parts[2] if len(parts) > 2 else None
                if sub_action == "on":
                    device.turn_on()
                elif sub_action == "off":
                    device.turn_off()
                else:
                    print("Invalid action for light.")
            elif action == "set":
                sub_action = parts[2] if len(parts) > 2 else None
                value = int(parts[3]) if len(parts) > 3 else None
                if sub_action == "brightness" and value is not None:
                    device.set_brightness(value)
                else:
                    print("Invalid action or missing value for brightness.")
            else:
                print("Invalid action for light.")
        elif isinstance(device, Thermostat):
            if action == "set":
                sub_action = parts[2] if len(parts) > 2 else None
                value = int(parts[3]) if len(parts) > 3 else None
                if sub_action == "temperature" and value is not None:
                    device.set_temperature(value)
                else:
                    print("Invalid action or missing value for temperature.")
            else:
                print("Invalid action for thermostat.")
        else:
            print("Unsupported device type.")

# Example usage
home = HomeAutomationSystem()
living_room_light = Light("Living Room Light")
kitchen_thermostat = Thermostat("Kitchen Thermostat")

home.add_device(living_room_light)
home.add_device(kitchen_thermostat)

commands = [
    "turn on Living Room Light",
    "set brightness Living Room Light 75",
    "turn off Living Room Light",
    "set temperature Kitchen Thermostat 22"
]

for cmd in commands:
    home.process_command(cmd)
