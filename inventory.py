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

def main(scene):
    obj = scene.objects["YourObjectName"]  # Replace with your object name
    
    if "inventory" not in obj:
        obj["inventory"] = Inventory()
    
    inventory = obj["inventory"]

    # Example actions: Adding and removing items based on some conditions
    # Replace with appropriate input handling for bpy
    keyboard = bpy.context.window_manager.keyconfigs.active.keymaps['3D View'].keymap_items

    if keyboard['A'].active:
        item = Item(name="Potion", description="Restores 20 HP", quantity=1)
        inventory.add_item(item)
        print("Added Potion to inventory.")
    
    if keyboard['R'].active:
        inventory.remove_item("Potion", 1)
        print("Removed one Potion from inventory.")
    
    if keyboard['I'].active:
        print("Inventory:")
        print(inventory.display_inventory())

# Ensure this script runs continuously in the game engine
bpy.app.handlers.frame_change_post.append(main)
