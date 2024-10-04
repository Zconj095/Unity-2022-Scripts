class WorldSpace:
    def __init__(self):
        self.origin = Vector3(0, 0, 0)
        self.scale = Vector3(1, 1, 1)
        self.orientation = Quaternion(1, 0, 0, 0)

    def get_origin(self):
        return self.origin

    def get_scale(self):
        return self.scale

    def get_orientation(self):
        return self.orientation