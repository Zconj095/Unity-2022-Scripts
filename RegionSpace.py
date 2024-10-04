class RegionSpace:
    def __init__(self, world_space):
        self.world_space = world_space
        self.origin = Vector3(0, 0, 0)
        self.scale = Vector3(1, 1, 1)
        self.orientation = Quaternion(1, 0, 0, 0)

    def get_origin(self):
        return self.origin + self.world_space.get_origin()

    def get_scale(self):
        return self.scale * self.world_space.get_scale()

    def get_orientation(self):
        return self.orientation * self.world_space.get_orientation()