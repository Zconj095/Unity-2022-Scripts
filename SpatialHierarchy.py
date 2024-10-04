class SpatialHierarchy:
    def __init__(self):
        self.world_space = WorldSpace()
        self.regional_spaces = {}
        self.level_spaces = {}

    def add_region(self, region_name, region_space):
        self.regional_spaces[region_name] = region_space

    def add_level(self, level_name, level_space):
        self.level_spaces[level_name] = level_space

    def get_world_space(self):
        return self.world_space

    def get_region_space(self, region_name):
        return self.regional_spaces[region_name]

    def get_level_space(self, level_name):
        return self.level_spaces[level_name]