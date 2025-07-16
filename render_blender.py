import argparse
import sys
import os
import math
import bpy
from mathutils import Vector
import numpy as np
import json

def init_render(engine='CYCLES', resolution=512, samples=64):
    """Configures Blender's rendering engine settings."""
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True
    
    if engine == 'CYCLES':
        bpy.context.scene.cycles.samples = samples
        bpy.context.scene.cycles.use_denoising = True
        try:
            bpy.context.scene.cycles.device = 'GPU'
            bpy.context.preferences.addons['cycles'].preferences.get_devices()
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        except Exception as e:
            print(f"  [Blender] WARNING: Could not set Cycles to GPU/CUDA. Error: {e}")
    elif engine == 'BLENDER_EEVEE':
        bpy.context.scene.eevee.taa_render_samples = samples

def init_scene():
    """Clears the default Blender scene."""
    if bpy.context.object and bpy.context.object.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.images]:
        for block in collection:
            collection.remove(block)

def init_camera():
    """Creates and configures a new camera."""
    cam_data = bpy.data.cameras.new('Camera')
    cam = bpy.data.objects.new('Camera', cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    track_to_empty = bpy.data.objects.new("TrackToEmpty", None)
    bpy.context.scene.collection.objects.link(track_to_empty)
    cam_constraint.target = track_to_empty
    return cam

def init_lighting():
    """Creates a standard three-point lighting setup."""
    bpy.data.worlds['World'].use_nodes = True
    bpy.data.worlds['World'].node_tree.nodes['Background'].inputs['Strength'].default_value = 0.5
    key_light_data = bpy.data.lights.new(name="KeyLight", type='AREA')
    key_light_data.energy = 5000
    key_light = bpy.data.objects.new(name="KeyLight", object_data=key_light_data)
    bpy.context.scene.collection.objects.link(key_light)
    key_light.location = (4, 4, 5)

def load_object(object_path: str):
    """Loads a .glb or .obj file."""
    file_extension = object_path.split(".")[-1].lower()
    if file_extension in {"glb", "gltf"}:
        bpy.ops.import_scene.gltf(filepath=object_path)
    elif file_extension == 'obj':
        bpy.ops.import_scene.obj(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def get_scene_meshes():
    return [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

def scene_bbox(meshes):
    bbox_min = Vector((math.inf, math.inf, math.inf))
    bbox_max = Vector((-math.inf, -math.inf, -math.inf))
    if not meshes: return None, None
    for mesh_obj in meshes:
        for corner in mesh_obj.bound_box:
            world_corner = mesh_obj.matrix_world @ Vector(corner)
            bbox_min = Vector(min(a, b) for a, b in zip(bbox_min, world_corner))
            bbox_max = Vector(max(a, b) for a, b in zip(bbox_max, world_corner))
    return bbox_min, bbox_max

def normalize_scene():
    """Normalizes all mesh objects to fit within a unit cube."""
    meshes = get_scene_meshes()
    if not meshes: return
    bbox_min, bbox_max = scene_bbox(meshes)
    if bbox_min is None: return
    scale = 1.5 / max((bbox_max - bbox_min))
    offset = -(bbox_min + bbox_max) / 2.0
    root_objects = [obj for obj in bpy.context.scene.objects if obj.parent is None]
    parent_obj = root_objects[0]
    if len(root_objects) > 1:
        parent_obj = bpy.data.objects.new("NormalizationParent", None)
        bpy.context.scene.collection.objects.link(parent_obj)
        for obj in root_objects:
            obj.parent = parent_obj
    parent_obj.scale = (scale, scale, scale)
    parent_obj.location = offset * scale
    bpy.context.view_layer.update()

def main(args):
    print("[Blender Script Started]")
    init_scene()
    load_object(args.object)
    normalize_scene()
    
    if args.quality == 'FAST':
        engine = 'BLENDER_EEVEE'
        samples = 4
    else: # HIGH quality
        engine = 'CYCLES'
        samples = 64
        
    init_render(engine=engine, resolution=args.resolution, samples=samples)
    cam = init_camera()
    init_lighting()
    
    os.makedirs(args.output_folder, exist_ok=True)
    views = json.loads(args.views)
    print(f"  [Blender] Preparing to render {len(views)} views...")
    
    # --- MODIFIED PART: Internal loop for rendering all views ---
    for i, view in enumerate(views):
        radius = view.get('radius', 2.5)
        x = radius * np.cos(view['yaw']) * np.cos(view['pitch'])
        y = radius * np.sin(view['yaw']) * np.cos(view['pitch'])
        z = radius * np.sin(view['pitch'])
        cam.location = (x, y, z)
        cam.data.angle = view['fov']
        
        # Set the output path for this specific view
        bpy.context.scene.render.filepath = os.path.join(args.output_folder, f'{i:03d}.png')
        
        # Render the single image
        bpy.ops.render.render(write_still=True)
    # --- END OF MODIFIED PART ---
    
    print("[Blender Script Finished Successfully]")

if __name__ == '__main__':
    try:
        argv = sys.argv[sys.argv.index("--") + 1:]
    except ValueError:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--views', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--quality', type=str, default='HIGH', choices=['FAST', 'HIGH'])
    args = parser.parse_args(argv)
    main(args)