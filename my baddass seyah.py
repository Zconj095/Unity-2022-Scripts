import bpy
import bmesh
import random
import numpy as np
import pickle

class TerrainOperator(bpy.types.Operator):
    bl_idname = "object.generate_terrain"
    bl_label = "Generate Terrain"
    bl_options = {'REGISTER', 'UNDO'}

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
        paint_terrain(obj, height_map)

        # Save the terrain
        save_height_map(height_map, "/path/to/save/height_map.pkl")

        return {'FINISHED'}

    def create_terrain(self, mesh):
        # Create a bmesh object
        bm = bmesh.new()

        # Terrain dimensions
        size = 50
        height = 5

        # Create a height map using Perlin noise
        scale = 10.0
        height_map = self.generate_height_map(size, scale, height)

        # Add random features
        height_map = add_hills(height_map, num_hills=5, hill_height=3)
        height_map = add_valleys(height_map, num_valleys=5, valley_depth=2)

        # Smooth the terrain
        height_map = smooth_height_map(height_map, iterations=2)

        # Create vertices
        for x in range(size + 1):
            for y in range(size + 1):
                z = height_map[x, y]
                bm.verts.new((x, y, z))

        bm.verts.ensure_lookup_table()

        # Create faces
        for x in range(size):
            for y in range(size):
                v1 = bm.verts[x * (size + 1) + y]
                v2 = bm.verts[x * (size + 1) + y + 1]
                v3 = bm.verts[(x + 1) * (size + 1) + y + 1]
                v4 = bm.verts[(x + 1) * (size + 1) + y]
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

    def add_material(self, obj):
        # Create a new material
        mat = bpy.data.materials.new(name="TerrainMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        
        # Add a noise texture
        noise_texture = mat.node_tree.nodes.new('ShaderNodeTexNoise')
        noise_texture.inputs['Scale'].default_value = 20.0
        
        # Connect the noise texture to the base color
        mat.node_tree.links.new(bsdf.inputs['Base Color'], noise_texture.outputs['Color'])
        
        # Assign the material to the object
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

def smooth_height_map(height_map, iterations=1):
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

def add_hills(height_map, num_hills=10, hill_height=2):
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

def add_valleys(height_map, num_valleys=10, valley_depth=2):
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

def paint_terrain(obj, height_map):
    mat_low = create_material("Low", (0.1, 0.5, 0.1, 1))  # Green for low areas
    mat_high = create_material("High", (0.5, 0.5, 0.5, 1))  # Gray for high areas

    size = height_map.shape[0]
    for x in range(size):
        for y in range(size):
            z = height_map[x, y]
            if z < 2:  # Threshold for low and high areas
                obj.data.vertices[x * size + y].material_index = 0
            else:
                obj.data.vertices[x * size + y].material_index = 1

def create_material(name, color):
    mat = bpy.data.materials.new(name=name)
    mat.diffuse_color = color
    return mat

def save_height_map(height_map, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(height_map, file)

def load_height_map(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)

def menu_func(self, context):
    self.layout.operator(TerrainOperator.bl_idname)

def register():
    bpy.utils.register_class(TerrainOperator)
    bpy.types.VIEW3D_MT_mesh_add.append(menu_func)

def unregister():
    bpy.utils.unregister_class(TerrainOperator)
    bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)

if __name__ == "__main__":
    register()
