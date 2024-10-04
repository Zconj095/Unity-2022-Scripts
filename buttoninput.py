def handle_actions(actions, obj):
    inventory = obj["inventory"]

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

    if actions["R3"]:
        print("Sprint")
        # Implement sprint action

    if actions["R2"]:
        print("Use item")
        # Implement using item

    if actions["Select"]:
        print("Open stats/skills menu")
        # Implement opening stats/skills menu

    # Handle inventory display separately
    keyboard = bpy.context.window_manager.keyconfigs.active.keymaps['3D View'].keymap_items

    if keyboard['I'].active:
        print("Inventory:")
        print(inventory.display_inventory())
