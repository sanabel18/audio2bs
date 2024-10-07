import bpy
from bpy.props import (StringProperty,
                       BoolProperty,
                       IntProperty,
                       FloatProperty,
                       EnumProperty,
                       PointerProperty,
                       )
from bpy.types import (Panel,
                       Operator,
                       PropertyGroup,
                       )
import json 
from bpy import context
import builtins as __builtin__

def console_print(*args, **kwargs):
    for a in context.screen.areas:
        if a.type == 'CONSOLE':
            c = {}
            c['area'] = a
            c['space_data'] = a.spaces.active
            c['region'] = a.regions[-1]
            c['window'] = context.window
            c['screen'] = context.screen
            s = " ".join([str(arg) for arg in args])
            for line in s.split("\n"):
                bpy.ops.console.scrollback_append(c, text=line)

def print(*args, **kwargs):
    """Console print() function."""

    console_print(*args, **kwargs) # to py consoles
    __builtin__.print(*args, **kwargs) # to system console


# ------------------------------------------------------------------------
#    Scene Properties
# ------------------------------------------------------------------------

#class JSONimportPanelProps(PropertyGroup):
#    filepath = bpy.props.StringProperty(name="String Value")
 

# ------------------------------------------------------------------------
#    Operators
# ------------------------------------------------------------------------        
        


class createAnimation(Operator):
    """Usage: select multiple blendshapes (face, upper/lower theeth, jaw) and then import 
              blendshape json file
       whenever a json file was imported, the old timeline keyframes will be deleted.
    """
    bl_label = "Create Animation"
    bl_idname = 'object.blendshape_animation_operator'
    
    filepath : bpy.props.StringProperty(name="JSON file", description="JSON blendshape file from Audio2Face",subtype="FILE_PATH") 
    def execute(self,context):
        display = "filepath= "+self.filepath  
        with open(self.filepath) as f:
            json_animate = json.load(f) 
            bs_animation = json_animate['weightMat']
            bs_pose_count = json_animate['numPoses']
            bs_frame_count = json_animate['numFrames']
            bs_names = json_animate['facsNames']
            bs_offset = 1 # basis blendshape at start
            bs_limit = bs_pose_count

            N = self.keyframe_num()
            self.delete_keyframe(N, bs_limit, bs_names, bs_animation)
            print('bs_frame_count {}'.format(bs_frame_count))
            for i in range(bs_frame_count): #each keyframe
                for j in range(bs_limit):#bs_pose_count):# each pose
                     num_objs = len(bpy.context.selected_objects)
                     for k in range(num_objs):
                        index = bpy.context.selected_objects[k].data.shape_keys.key_blocks.find(bs_names[j])
                        if index > -1: 
                            bpy.context.selected_objects[k].active_shape_key_index = index
                            bpy.context.selected_objects[k].active_shape_key.value = bs_animation[i][j]
                            bpy.context.selected_objects[k].active_shape_key.keyframe_insert("value",frame=i)
       return {'FINISHED'}
    
    def delete_keyframe(self, N, bs_limit, bs_names, bs_animation):
        for i in range(N): #each keyframe
            print(" delete frame: {} {}".format(i, N ));
            for j in range(bs_limit):#bs_pose_count):# each pose
                num_objs = len(bpy.context.selected_objects)
                for k in range(num_objs):
                    index = bpy.context.selected_objects[k].data.shape_keys.key_blocks.find(bs_names[j])
                    if index > -1: 
                        bpy.context.selected_objects[k].active_shape_key_index = index
                        bpy.context.selected_objects[k].active_shape_key.value = 0.0
                        bpy.context.selected_objects[k].active_shape_key.keyframe_delete("value",frame=i)
            
    def keyframe_num(self):
        bpy.ops.screen.frame_jump(end=False)
        n_forw=0

        # Counting forward:
        while True:
            ret=bpy.ops.screen.keyframe_jump(next=True)
            if ret!={'FINISHED'}:
                break
            else:
                n_forw+=1

        # Go to last frame:
        bpy.ops.screen.frame_jump(end=True)
        n_back=0

        # Counting backward:
        while True:
            ret=bpy.ops.screen.keyframe_jump(next=False)
            if ret!={'FINISHED'}:
                break
            else:
                n_back+=1

         # Max of forward and backward:
        N_keyframes=max(n_forw,n_back)
        print("N_keyframes {} {} {}".format(N_keyframes, n_forw, n_back))
        return N_keyframes
       
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self) 
        return {'RUNNING_MODAL'}  

# ------------------------------------------------------------------------
#    Panel in Object Mode
# ------------------------------------------------------------------------

class JSONimportPanel(Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Import A2F Blendshapes"
    bl_idname = "OBJECT_PT_hello"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Josh Tools'

    def draw(self, context):
        layout = self.layout
        
        obj = context.object
        
        row = layout.row()
        row.label(text="Select Object with face Blendshapes")

        row = layout.row()
        row.label(text="Active object is: " + obj.name)
                        
        row = layout.row()
        row.operator("object.blendshape_animation_operator")
        


def register():
    #bpy.utils.register_class(JSONimportPanelProps)
    bpy.utils.register_class(JSONimportPanel)
    bpy.utils.register_class(createAnimation)

def unregister():
    #bpy.utils.unregister_class(JSONimportPanelProps)
    bpy.utils.unregister_class(JSONimportPanel)
    bpy.utils.unregister_class(createAnimation)


if __name__ == "__main__":
    register()
