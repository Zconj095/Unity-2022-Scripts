class Player(GameObject):
    def __init__(self, scene):
        super().__init__(scene)
        self.speed = 10

    def update(self, delta_time):
        if keyboard.is_pressed('w'):
            self.position += Vector3(0, 0, self.speed * delta_time)
        if keyboard.is_pressed('s'):
            self.position -= Vector3(0, 0, self.speed * delta_time)
        if keyboard.is_pressed('a'):
            self.position -= Vector3(self.speed * delta_time, 0, 0)
        if keyboard.is_pressed('d'):
            self.position += Vector3(self.speed * delta_time, 0, 0)