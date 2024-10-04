import bpy
import bmesh
import random
import numpy as np
import math

class TerrainGenerator(bpy.types.Operator):
    bl_idname = "mesh.generate_terrain"
    bl_label = "Generate Terrain"
    bl_options = {'REGISTER', 'UNDO'}

    size: bpy.props.IntProperty(name="Size", default=50, min=10, max=1000)
    height: bpy.props.FloatProperty(name="Height", default=5.0, min=1.0, max=20.0)
    num_hills: bpy.props.IntProperty(name="Number of Hills", default=10, min=1, max=50)
    hill_height: bpy.props.FloatProperty(name="Hill Height", default=2.0, min=0.1, max=10.0)
    num_valleys: bpy.props.IntProperty(name="Number of Valleys", default=10, min=1, max=50)
    valley_depth: bpy.props.FloatProperty(name="Valley Depth", default=2.0, min=0.1, max=10.0)
    num_rivers: bpy.props.IntProperty(name="Number of Rivers", default=2, min=1, max=10)
    river_depth: bpy.props.FloatProperty(name="River Depth", default=1.0, min=0.1, max=5.0)
    num_mountains: bpy.props.IntProperty(name="Number of Mountains", default=2, min=1, max=10)
    peak_height: bpy.props.FloatProperty(name="Peak Height", default=10.0, min=1.0, max=20.0)
    erosion_iterations: bpy.props.IntProperty(name="Erosion Iterations", default=100, min=10, max=500)
    smoothing_iterations: bpy.props.IntProperty(name="Smoothing Iterations", default=2, min=1, max=10)

    def execute(self, context):
        # Create a new mesh and object
        mesh = bpy.data.meshes.new("TerrainMesh")
        obj = bpy.data.objects.new("Terrain", mesh)

        # Link the object to the scene
        bpy.context.collection.objects.link(obj)

        # Select the object
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # Create the terrain
        height_map = self.create_terrain(mesh)
        self.add_material(obj)

        # Paint the terrain
        self.paint_terrain(obj, height_map)

        return {'FINISHED'}

    def create_terrain(self, mesh):
        # Create a bmesh object
        bm = bmesh.new()

        # Create a height map using Perlin noise
        scale = 10.0
        height_map = self.generate_height_map(self.size, scale, self.height)

        # Add random features
        height_map = self.add_hills(height_map, num_hills=self.num_hills, hill_height=self.hill_height)
        height_map = self.add_valleys(height_map, num_valleys=self.num_valleys, valley_depth=self.valley_depth)
        height_map = self.add_rivers(height_map, num_rivers=self.num_rivers, river_depth=self.river_depth)
        height_map = self.add_mountains(height_map, num_mountains=self.num_mountains, peak_height=self.peak_height)

        # Simulate erosion
        height_map = self.simulate_erosion(height_map, iterations=self.erosion_iterations)

        # Smooth the terrain
        height_map = self.smooth_height_map(height_map, iterations=self.smoothing_iterations)

        # Create vertices
        for x in range(self.size + 1):
            for y in range(self.size + 1):
                z = height_map[x, y]
                bm.verts.new((x, y, z))

        bm.verts.ensure_lookup_table()

        # Create faces
        for x in range(self.size):
            for y in range(self.size):
                v1 = bm.verts[x * (self.size + 1) + y]
                v2 = bm.verts[x * (self.size + 1) + y + 1]
                v3 = bm.verts[(x + 1) * (self.size + 1) + y + 1]
                v4 = bm.verts[(x + 1) * (self.size + 1) + y]
                bm.faces.new((v1, v2, v3, v4))

        # Finish up, write the bmesh into the mesh
        bm.to_mesh(mesh)
        bm.free()

        return height_map

    def generate_height_map(self, size, scale, height):
        height_map = np.zeros((size + 1, size + 1))

        for x in range(size + 1):
            for y in range(size + 1):
                height_map[x, y] = random.uniform(0, height) * np.sin(x / scale) * np.cos(y / scale)

        return height_map

    def add_hills(self, height_map, num_hills=10, hill_height=2):
        size = height_map.shape[0]
        for _ in range(num_hills):
            hill_x = random.randint(0, size - 1)
            hill_y = random.randint(0, size - 1)
            radius = random.randint(1, size // 10)
            for x in range(hill_x - radius, hill_x + radius):
                for y in range(hill_y - radius, hill_y + radius):
                    if 0 <= x < size and 0 <= y < size:
                        distance = ((hill_x - x) ** 2 + (hill_y - y) ** 2) ** 0.5
                        if distance < radius:
                            height_map[x, y] += hill_height * (1 - distance / radius)
        return height_map

    def add_valleys(self, height_map, num_valleys=10, valley_depth=2):
        size = height_map.shape[0]
        for _ in range(num_valleys):
            valley_x = random.randint(0, size - 1)
            valley_y = random.randint(0, size - 1)
            radius = random.randint(1, size // 10)
            for x in range(valley_x - radius, valley_x + radius):
                for y in range(valley_y - radius, valley_y + radius):
                    if 0 <= x < size and 0 <= y < size:
                        distance = ((valley_x - x) ** 2 + (valley_y - y) ** 2) ** 0.5
                        if distance < radius:
                            height_map[x, y] -= valley_depth * (1 - distance / radius)
        return height_map

    def add_rivers(self, height_map, num_rivers=2, river_depth=1):
        size = height_map.shape[0]
        for _ in range(num_rivers):
            x, y = random.randint(0, size - 1), random.randint(0, size - 1)
            for _ in range(size // 2):
                height_map[x, y] -= river_depth
                direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                x, y = (x + direction[0]) % size, (y + direction[1]) % size
        return height_map

    def add_mountains(self, height_map, num_mountains=2, peak_height=10):
        size = height_map.shape[0]
        for _ in range(num_mountains):
            peak_x = random.randint(0, size - 1)
            peak_y = random.randint(0, size - 1)
            for x in range(size):
                for y in range(size):
                    distance = math.sqrt((peak_x - x) ** 2 + (peak_y - y) ** 2)
                    height_map[x, y] += peak_height * math.exp(-distance / 10)
        return height_map

    def simulate_erosion(self, height_map, iterations=100):
        size = height_map.shape[0]
        for _ in range(iterations):
            x, y = random.randint(1, size - 2), random.randint(1, size - 2)
            sediment = height_map[x, y] * 0.1
            for _ in range(20):
                height_map[x, y] -= sediment
                direction = np.array([height_map[x-1, y] - height_map[x+1, y], height_map[x, y-1] - height_map[x, y+1]])
                direction = direction / np.linalg.norm(direction)
                x = min(max(int(x + direction[0]), 1), size - 2)
                y = min(max(int(y + direction[1]), 1), size - 2)
                height_map[x, y] += sediment
        return height_map

    def smooth_height_map(self, height_map, iterations=1):
        for _ in range(iterations):
            smoothed_map = height_map.copy()
            for x in range(1, height_map.shape[0] - 1):
                for y in range(1, height_map.shape[1] - 1):
                    smoothed_map[x, y] = (
                        height_map[x, y] +
                        height_map[x-1, y] +
                        height_map[x+1, y] +
                        height_map[x, y-1] +
                        height_map[x, y+1]
                    ) / 5.0
            height_map = smoothed_map
        return height_map

    def add_material(self, obj):
        # Create new materials
        mat_low = bpy.data.materials.new(name="Low")
        mat_low.diffuse_color = (0.1, 0.5, 0.1, 1)  # Green for low areas

        mat_high = bpy.data.materials.new(name="High")
        mat_high.diffuse_color = (0.5, 0.5, 0.5, 1)  # Gray for high areas

        obj.data.materials.append(mat_low)
        obj.data.materials.append(mat_high)

    def paint_terrain(self, obj, height_map):
        size = height_map.shape[0]
        obj.data.polygons.foreach_set("material_index", [0] * len(obj.data.polygons))  # Default to low material

        for face in obj.data.polygons:
            vert_indices = face.vertices
            heights = [height_map[int(obj.data.vertices[v].co.x)][int(obj.data.vertices[v].co.y)] for v in vert_indices]
            average_height = sum(heights) / len(heights)

            if average_height >= 2:  # Threshold for high and low material
                face.material_index = 1

def menu_func(self, context):
    self.layout.operator(TerrainGenerator.bl_idname)

def register():
    bpy.utils.register_class(TerrainGenerator)
    bpy.types.VIEW3D_MT_mesh_add.append(menu_func)

def unregister():
    bpy.utils.unregister_class(TerrainGenerator)
    bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)

if __name__ == "__main__":
    register()
