class Camera:
    def __init__(self, name, position, rotation, zoom):
        self.name = name
        self.position = position
        self.rotation = rotation
        self.zoom = zoom

    def update(self, delta_time):
        # Update the camera's state based on delta time
        pass

    def render(self, scene):
        # Render the scene using the camera
        pass

    def handle_input(self, input_event):
        # Handle user input events
        pass

    def get_position(self):
        return self.position

    def get_rotation(self):
        return self.rotation

    def get_zoom(self):
        return self.zoom