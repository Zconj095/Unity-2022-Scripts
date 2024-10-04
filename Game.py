class Game:
    def __init__(self, scene_loader):
        self.scene_loader = scene_loader
        self.scene_manager = SceneManager(scene_loader)
        self.current_scene = None

    def run(self):
        while True:
            # Handle user input
            input_event = get_input_event()
            self.scene_manager.handle_input(input_event)

            # Update the game state
            delta_time = get_delta_time()
            if self.current_scene:
                self.current_scene.update(delta_time)

            # Render the game scene
            if self.current_scene:
                self.current_scene.render()

            # Swap buffers to display the new frame
            swap_buffers()