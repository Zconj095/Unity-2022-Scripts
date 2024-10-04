import bpy

def paint_texture(texture_path, obj):
    img = bpy.data.images.load(texture_path)
    tex = bpy.data.textures.new(name="TerrainTexture", type='IMAGE')
    tex.image = img
    mat = bpy.data.materials.new(name="TerrainMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    tex_node = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex_node.image = img
    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_node.outputs['Color'])
    obj.data.materials.append(mat)
