import bpy
from math import radians
from mathutils import Vector
import numpy as np
import cv2
import os
from time import time 
import math
import json

with open('blender_config.json', 'r') as f:
    config = json.load(f)


def whiten_model(obj):
    white_material = bpy.data.materials.new(name="WhiteMaterial")
    white_material.use_nodes = True
    nodes = white_material.node_tree.nodes
    links = white_material.node_tree.links
    for node in nodes:
        nodes.remove(node)
    diffuse_node = nodes.new(type='ShaderNodeBsdfDiffuse')
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(diffuse_node.outputs['BSDF'], output_node.inputs['Surface'])
    if len(obj.data.materials) > 0:
        obj.data.materials[0] = white_material
    else:
        obj.data.materials.append(white_material)

def calculate_angle(height, base_length):
    if base_length == 0:
        raise ValueError("Base length cannot be zero.")
    angle_radians = math.atan(height / base_length)
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees

start = time()

person_x, person_y, person_z = config['person_location']
camera_x, camera_y, camera_z = config['camera_location']

# theta_x = 50
# theta_y = 0
# theta_z = 160
for theta_z in [160]:
    keyword = config['keyword']
    GEN_MODEL = f'/Users/gauranshurathee/Desktop/shadow_positioning/test_data/nov26/obj/delighted.obj'
    result_path = config['output_path'].replace("KEYWORD", keyword)
    # keyword = 'image11'
    # GEN_MODEL = f'/Users/gauranshurathee/Desktop/shadow_positioning/test_data/nov26/obj/delighted.obj'
    # result_path = f"{keyword}_topview"

    # Set render engine
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'

    scene.cycles.samples = config['cycle_samples']
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = True

    # Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Create a white background image
    white_bg = np.ones((1000, 1000, 3)).astype('uint8') * 255
    white_image_path = config['background']['image_path']
    cv2.imwrite(white_image_path, white_bg)

    bg_image = bpy.data.images.load(white_image_path)
    width, height = bg_image.size

    # Desired maximum dimension (1K resolution)
    max_dimension = config['background']['resolution']

    # Calculate aspect ratio
    aspect_ratio = width / height

    # Determine render resolution while maintaining aspect ratio
    if width >= height:
        render_width = max_dimension
        render_height = int(max_dimension / aspect_ratio)
    else:
        render_height = max_dimension
        render_width = int(max_dimension * aspect_ratio)

    # Set render resolution to 1K while maintaining aspect ratio
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    scene.render.resolution_percentage = 102  # Ensure full resolution

    # Import the 3D model
    bpy.ops.wm.obj_import(filepath=GEN_MODEL)
    person = bpy.context.selected_objects[0]
    whiten_model(person)
    pl_x, pl_y, pl_z = config['person_location']
    person.location = Vector((pl_x, pl_y, pl_z))  # Adjust Z if needed
    person.scale = config['person_scale']
    # person.scale = (1, 1, 1)  # Adjust as necessary
    person.is_shadow_catcher = False
    person.is_holdout = False

    angle_z = calculate_angle(person_y-camera_y, person_x-camera_x)
    # pr_x, pr_y, pr_z = config['person_rotation']
    # person.rotation_euler = (radians(90), radians(0), radians(0))

    pr_x, pr_y, pr_z = config['person_rotation']
    # person.rotation_euler = (radians(pr_x), radians(pr_y), radians(angle_z-180*(angle_z>0)+90))
    person.rotation_euler = (radians(pr_x), radians(pr_y), radians(pr_z))
    # person.rotation_euler = (radians(90), radians(0), radians(angle_z-180*(angle_z>0)+90))

    # Add plane as shadow catcher
    bpy.ops.mesh.primitive_plane_add(size=config['plane_size'])
    plane = bpy.context.active_object
    plane.location = Vector((pl_x, pl_y, pl_z-0.9))
    # plane.rotation = 
    # plane.location = Vector((0, 9.65, -1.955))
    plane.is_shadow_catcher = True

    # Set up lighting
    bpy.ops.object.light_add(type='SUN')
    sun = bpy.context.active_object

    sr_x, sr_y, sr_z = config['light_rotation']
    sun.rotation_euler = (radians(sr_x), radians(sr_y), radians(sr_z))  
    sun.location = Vector(config['light_location'])  # Adjust as needed
    # 
    # Adjust as needed
    sun.data.energy = 10

    # Add a camera
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    scene.camera = camera

    # Position and rotate the camera to face the scene front-on
    camera.location = Vector(config['camera_location'])  # Move along -Y axis
    cr_x, cr_y, cr_z = config['camera_rotation']
    camera.rotation_euler = (radians(cr_x), radians(cr_y), radians(cr_z))  # Facing along +Y axis

    # Set camera focal length
    camera.data.lens = config['camera_lens']  # Adjust based on background image

    # Enable background image in camera view
    camera.data.show_background_images = True
    bg = camera.data.background_images.new()
    bg.image = bg_image
    bg.alpha = 1.0
    bg.display_depth = 'BACK'

    # Set camera clipping planes
    camera.data.clip_start = config['camera_clip_start']
    camera.data.clip_end = config['camera_end']

    os.makedirs('temp', exist_ok=True)

    # Set render settings
    scene.render.film_transparent = True
    scene.render.filepath = f"shadow_info"

    # Use compositing nodes to overlay the background image
    scene.use_nodes = True
    tree = scene.node_tree
    links = tree.links

    # Clear existing nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Create new nodes
    render_layers = tree.nodes.new(type='CompositorNodeRLayers')
    bg_node = tree.nodes.new(type='CompositorNodeImage')
    bg_node.image = bg_image
    scale_node = tree.nodes.new(type='CompositorNodeScale')
    scale_node.space = 'RENDER_SIZE'
    alpha_over = tree.nodes.new(type='CompositorNodeAlphaOver')
    composite = tree.nodes.new(type='CompositorNodeComposite')
    
    # scene.view_settings.view_transform = 'Standard' 
    

    # Connect nodes
    links.new(bg_node.outputs['Image'], scale_node.inputs['Image'])
    links.new(scale_node.outputs['Image'], alpha_over.inputs[1])        # Background (scaled)
    links.new(render_layers.outputs['Image'], alpha_over.inputs[2])     # Foreground
    links.new(alpha_over.outputs['Image'], composite.inputs['Image'])

    # Render the scene with the shadow
    # bpy.ops.render.render(write_still=True)

    # -------------------------
    # Additional section to save the mask of the person
    # -------------------------

    # Print available view layer names to check the correct name
    print("Available view layers:", [layer.name for layer in scene.view_layers])

    # Get the view layer by name, fallback to the first view layer if "View Layer" is not found
    view_layer = scene.view_layers.get("View Layer", scene.view_layers[0])

    # Enable object index pass
    view_layer.use_pass_object_index = True  # Enable object index pass

    # Set the person object index
    person.pass_index = 1  # Assign a unique pass index to the person

    # Create mask render settings
    mask_file_path = result_path

    # Add mask output nodes to the compositor
    mask_output = tree.nodes.new(type="CompositorNodeOutputFile")
    mask_output.base_path = ''  # Use base_path if you want all images in a common folder
    mask_output.file_slots[0].path = mask_file_path
    mask_output.format.file_format = 'JPEG'

    # Add an ID Mask node to isolate the person
    id_mask_node = tree.nodes.new(type="CompositorNodeIDMask")
    id_mask_node.index = 1  # Match the person's pass index

    # Connect nodes for mask output
    links.new(render_layers.outputs['IndexOB'], id_mask_node.inputs['ID value'])  # Object index output to ID Mask
    links.new(id_mask_node.outputs['Alpha'], mask_output.inputs[0])  # Connect mask to output file

    # Perform the second render for the mask
    scene.render.filepath = mask_file_path  # Set output file path for mask render
    bpy.ops.render.render(write_still=True)

    # -------------------------

    end = time()
    print(f'Total Time Taken: {end-start}')
