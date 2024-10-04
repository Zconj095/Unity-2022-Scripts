class LevelSpace:
    def __init__(self, region_space):
        self.region_space = region_space
        self.origin = Vector3(0, 0, 0)
        self.scale = Vector3(1, 1, 1)
        self.orientation = Quaternion(1, 0, 0, 0)

    def get_origin(self):
        return self.origin + self.region_space.get_origin()

    def get_scale(self):
        return self.scale * self.region_space.get_scale()

    def get_orientation(self):
        return self.orientation * self.region_space.get_orientation()