class Space:
    def __init__(self, world_space, is_public=False, is_private=True, is_personal=True):
        self.world_space = world_space
        self.is_public = is_public
        self.is_private = is_private
        self.is_personal = is_personal

    def get_world_space(self):
        return self.world_space

    def set_world_space(self, world_space):
        self.world_space = world_space

class LocalSpace(Space):
    def __init__(self, region_space, is_public=False, is_private=True, is_personal=True):
        super().__init__(region_space, is_public, is_private, is_personal)
        self.region_space = region_space

    def get_region_space(self):
        return self.region_space

    def set_region_space(self, region_space):
        self.region_space = region_space

class RegionSpace(Space):
    def __init__(self, world_space, is_public=False, is_private=True, is_personal=True):
        super().__init__(world_space, is_public, is_private, is_personal)
        self.world_space = world_space

    def get_world_space(self):
        return self.world_space

    def set_world_space(self, world_space):
        self.world_space = world_space

class LevelSpace(Space):
    def __init__(self, region_space, is_public=False, is_private=True, is_personal=True):
        super().__init__(region_space, is_public, is_private, is_personal)
        self.region_space = region_space

    def get_region_space(self):
        return self.region_space

    def set_region_space(self, region_space):
        self.region_space = region_space

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

class GameObject:
    def __init__(self, level_space, is_public=False, is_private=True, is_personal=True):
        self.level_space = level_space
        self.is_public = is_public
        self.is_private = is_private
        self.is_personal = is_personal

    def get_level_space(self):
        return self.level_space

    def set_level_space(self, level_space):
        self.level_space = level_space
        
world_space = WorldSpace()

region_space = RegionSpace(world_space)
level_space = LevelSpace(region_space)

game_object = GameObject(level_space, is_public=False, is_private=True, is_personal=True)