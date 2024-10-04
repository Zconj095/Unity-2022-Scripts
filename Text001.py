import bpy
import bmesh
import numpy as np
from mathutils import Vector
from PIL import Image

class SculptTIFF(bpy.types.Operator):
    bl_idname = "object.sculpt_tiff"
    bl_label = "Sculpt TIFF"
    bl_options = {'REGISTER', 'UNDO'}

    tiff_file: bpy.props.StringProperty(name="TIFF File", subtype='FILE_PATH')

    def execute(self, context):
        if not self.tiff_file:
            self.report({'ERROR'}, "No TIFF file selected")
            return {'CANCELLED'}

        # Load the TIFF file using Pillow
        try:
            img = Image.open(self.tiff_file)
            height_data = np.array(img)
        except Exception as e:
            self.report({'ERROR'}, f"Could not open TIFF file: {e}")
            return {'CANCELLED'}

        # Normalize height data
        min_height = height_data.min()
        max_height = height_data.max()
        height_data = (height_data - min_height) / (max_height - min_height)

        # Get the active object
        obj = context.object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "No valid mesh object selected")
            return {'CANCELLED'}

        # Ensure the object is in edit mode
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')

        # Get the bmesh representation of the mesh
        bm = bmesh.from_edit_mesh(obj.data)

        # Apply height data to vertices
        for vert in bm.verts:
            x = int((vert.co.x / obj.dimensions.x) * height_data.shape[1])
            y = int((vert.co.y / obj.dimensions.y) * height_data.shape[0])
            vert.co.z = height_data[y, x] * obj.dimensions.z

        # Update the mesh
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')

        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

def menu_func(self, context):
    self.layout.operator(SculptTIFF.bl_idname)

def register():
    bpy.utils.register_class(SculptTIFF)
    bpy.types.VIEW3D_MT_mesh_add.append(menu_func)

def unregister():
    bpy.utils.unregister_class(SculptTIFF)
    bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)

if __name__ == "__main__":
    register()
