class Item:
    def __init__(self, name, description, quantity):
        self.name = name
        self.description = description
        self.quantity = quantity

    def __str__(self):
        return f"{self.name} (x{self.quantity}): {self.description}"

class Inventory:
    def __init__(self):
        self.items = []

    def add_item(self, item):
        for inv_item in self.items:
            if inv_item.name == item.name:
                inv_item.quantity += item.quantity
                return
        self.items.append(item)

    def remove_item(self, item_name, quantity=1):
        for inv_item in self.items:
            if inv_item.name == item_name:
                inv_item.quantity -= quantity
                if inv_item.quantity <= 0:
                    self.items.remove(inv_item)
                return

    def get_item(self, item_name):
        for inv_item in self.items:
            if inv_item.name == item_name:
                return inv_item
        return None

    def display_inventory(self):
        inventory_list = [str(item) for item in self.items]
        return "\n".join(inventory_list)

import bpy

def init_joysticks():
    joysticks = [joy for joy in bpy.context.window_manager.gamepad_devices if joy is not None]
    
    if len(joysticks) == 0:
        print("No joysticks found.")
        return None, None
    
    if len(joysticks) == 1:
        print(f"One joystick found: {joysticks[0].name}")
        return joysticks[0], None
    
    print(f"Two joysticks found: {joysticks[0].name} and {joysticks[1].name}")
    return joysticks[0], joysticks[1]

def read_joystick(joystick):
    axes = joystick.axis_values
    buttons = joystick.button_values
    
    active_buttons = [i for i, value in enumerate(buttons) if value]
    
    return axes, active_buttons

def map_buttons(buttons):
    button_actions = {
        4: "L1",       # Left Bumper (L1)
        10: "R3",      # Right Stick Button (R3)
        6: "L2",       # Left Trigger (L2)
        7: "R2",       # Right Trigger (R2)
        8: "Start",    # Start Button
        9: "Select",   # Select Button
        0: "A",        # A Button
        1: "B",        # B Button
        2: "X",        # X Button
        3: "Y",        # Y Button
        5: "R1",       # Right Bumper (R1)
    }
    
    actions = {action: False for action in button_actions.values()}
    
    for button in buttons:
        if button in button_actions:
            actions[button_actions[button]] = True
    
    return actions

def handle_actions(actions, inventory):
    if actions["L1"]:
        print("Aim down scope")
        # Implement aiming down scope

    if actions["R3"]:
        print("Melee")
        # Implement melee attack

    if actions["L2"]:
        print("Use weapon")
        # Implement using weapon

    if actions["Start"]:
        print("Open menu")
        # Implement opening menu

    if actions["A"]:
        item = Item(name="Potion", description="Restores 20 HP", quantity=1)
        inventory.add_item(item)
        print("Added Potion to inventory.")

    if actions["B"]:
        inventory.remove_item("Potion", 1)
        print("Removed one Potion from inventory.")

    if actions["X"]:
        print("Attack")
        # Implement attack action

    if actions["Y"]:
        print("Jump")
        # Implement jump action

    if actions["R1"]:
        print("Block")
        # Implement blocking action

    if actions["R2"]:
        print("Use item")
        # Implement using item

    if actions["Select"]:
        print("Open stats/skills menu")
        # Implement opening stats/skills menu

    # Handle D-Pad input (axes)
    # Assuming D-Pad is mapped to axes 6 and 7 for horizontal and vertical movement
    if len(axes) > 6:
        dpad_horizontal = axes[6]
        dpad_vertical = axes[7]
        
        if dpad_horizontal < -0.5:
            print("Move left")
            # Implement moving left

        if dpad_horizontal > 0.5:
            print("Move right")
            # Implement moving right

        if dpad_vertical < -0.5:
            print("Move up")
            # Implement moving up

        if dpad_vertical > 0.5:
            print("Move down")
            # Implement moving down

def main(scene):
    obj = scene.objects["YourObjectName"]  # Replace with your object name
    
    if "inventory" not in obj:
        obj["inventory"] = Inventory()
    
    inventory = obj["inventory"]
    
    if "joystick1" not in obj:
        obj["joystick1"], obj["joystick2"] = init_joysticks()
    
    joystick1 = obj["joystick1"]
    joystick2 = obj["joystick2"]
    
    if joystick1:
        axes1, buttons1 = read_joystick(joystick1)
        actions1 = map_buttons(buttons1)
        handle_actions(actions1, inventory)
    
    if joystick2:
        axes2, buttons2 = read_joystick(joystick2)
        actions2 = map_buttons(buttons2)
        handle_actions(actions2, inventory)

# Ensure this script runs continuously in the game engine
bpy.app.handlers.frame_change_post.append(main)
