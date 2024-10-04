class GameObject:
    def __init__(self, name, position, rotation, scale):
        self.name = name
        self.position = position
        self.rotation = rotation
        self.scale = scale

    def update(self, delta_time):
        # Update the object's state based on delta time
        pass

    def render(self, camera):
        # Render the object using the camera
        pass

    def handle_input(self, input_event):
        # Handle user input events
        pass

    def get_position(self):
        return self.position

    def get_rotation(self):
        return self.rotation

    def get_scale(self):
        return self.scale