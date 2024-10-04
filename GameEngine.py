class GameEngine:
    def __init__(self):
        self.spatial_hierarchy = SpatialHierarchy()

    def add_region(self, region_name, region_space):
        self.spatial_hierarchy.add_region(region_name, region_space)

    def add_level(self, level_name, level_space):
        self.spatial_hierarchy.add_level(level_name, level_space)

    def get_world_space(self):
        return self.spatial_hierarchy.get_world_space()

    def get_region_space(self, region_name):
        return self.spatial_hierarchy.get_region_space(region_name)

    def get_level_space(self, level_name):
        return self.spatial_hierarchy.get_level_space(level_name)