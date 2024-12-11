import numpy as np
import bpy
from math import radians
from mathutils import Vector
import math
import json
from scipy.ndimage import label

HDRI_STRENGTH = 1.0
ARTIFICIAL_LIGHT_STRENGTH = 1.0
ARTIFICIAL_LIGHT_BOOST = 5.0 

# Load configuration
with open("blender_hdri_config.json", "r") as config_file:
    config = json.load(config_file)

# Variables from config
keyword = config["keyword"]
bg_image_path = config["bg_image_path"].replace("KEYWORD", keyword)
model_path = config["model_path"]
hdri_path = config["hdri_path"].replace('KEYWORD', keyword)
output_render_path = config["output_render_path"].replace('KEYWORD', keyword)

camera_position = Vector(config["camera_position"])
camera_rotation = tuple(radians(x) for x in config["camera_rotation"])
camera_lens = config["camera_lens"]

person_position = Vector(config["person_position"])
scaling_val = config["person_scale"]
person_scale = (scaling_val, scaling_val, scaling_val)
person_rotation = tuple(radians(x) for x in config["person_rotation"])

plane_position_offset = config["plane_position_offset"]

sr_x, sr_y, sr_z = (radians(x) for x in config["sun_rotation"])
sun_energy = config["sun_energy"]

max_dimension = config["max_dimension"]
render_samples = config["render_samples"]

def calculate_angle(height, base_length):
    if base_length == 0:
        raise ValueError("Base length cannot be zero.")
    angle_radians = math.atan(height / base_length)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

# Set render engine to Cycles
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = render_samples
scene.cycles.use_denoising = True
scene.cycles.use_adaptive_sampling = True

# Enable GPU rendering
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'NONE'
scene.cycles.device = 'CPU'

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Load background image and get dimensions
bg_image = bpy.data.images.load(bg_image_path)
width, height = bg_image.size
aspect_ratio = width / height
if width >= height:
    render_width = max_dimension
    render_height = int(max_dimension / aspect_ratio)
else:
    render_height = max_dimension
    render_width = int(max_dimension * aspect_ratio)

scene.render.resolution_x = render_width
scene.render.resolution_y = render_height
scene.render.resolution_percentage = 100

# Add a camera
bpy.ops.object.camera_add()
camera = bpy.context.active_object
scene.camera = camera
camera.location = camera_position
camera.rotation_euler = camera_rotation
camera.data.lens = camera_lens

# Import the 3D model
bpy.ops.wm.obj_import(filepath=model_path)
person = bpy.context.selected_objects[0]
bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')


pl_x, pl_y, pl_z = config['person_position']
pr_x, pr_y, pr_z = config['person_rotation']
cl_x, cl_y, cl_z = config['camera_position']
angle_z = calculate_angle(pl_y-cl_y, pl_x-cl_x)

person.location = person_position
person.scale = person_scale
# person.rotation_euler = (radians(pr_x), radians(pr_y), radians(0))
person.rotation_euler = (radians(pr_x), radians(pr_y), radians(angle_z-180*(angle_z>0)+90))


# Add plane as shadow catcher
bpy.ops.mesh.primitive_plane_add(size=50)
plane = bpy.context.active_object
plane.location = person_position + Vector((0, 0, plane_position_offset))
plane.rotation_euler = (radians(5), radians(0), radians(0))
plane.is_shadow_catcher = True

# Modify shadow catcher material to reduce shadow opacity
plane_material = bpy.data.materials.new(name="ShadowCatcherMaterial")
plane_material.use_nodes = True
nodes = plane_material.node_tree.nodes
links = plane_material.node_tree.links

# Clear existing nodes
for node in nodes:
    nodes.remove(node)

# Add nodes for shadow catcher material
output_node = nodes.new(type='ShaderNodeOutputMaterial')
mix_node = nodes.new(type='ShaderNodeMixShader')
transparent_node = nodes.new(type='ShaderNodeBsdfTransparent')
diffuse_node = nodes.new(type='ShaderNodeBsdfDiffuse')

# Connect nodes
links.new(transparent_node.outputs['BSDF'], mix_node.inputs[1])
links.new(diffuse_node.outputs['BSDF'], mix_node.inputs[2])
links.new(mix_node.outputs['Shader'], output_node.inputs['Surface'])

# Set shadow intensity (blend factor)
mix_node.inputs['Fac'].default_value = 0.93  # Lower values make shadows lighter

plane.data.materials.append(plane_material)

# ############################ sunlight ###############################

# # Commented out Sunlight
# bpy.ops.object.light_add(type='SUN')
# sun = bpy.context.active_object
# sun.rotation_euler = (sr_x, sr_y, sr_z)
# sun.location = Vector((0, 0, 20))
# sun.data.energy = sun_energy
# sun.data.angle = 0.1
# ############################ sunlight ###############################


# ############################ indoor lighting ###############################

# Define the grid size and spacing for the lights
num_lights_x = 8  # Number of lights along the X-axis
num_lights_y = 2  # Number of lights along the Y-axis
spacing_x = .2  # Spacing between lights along the X-axis
spacing_y = 0.5  # Spacing between lights along the Y-axis
height_z = 1  # Height of the lights (ceiling)

# # Add point lights in a grid pattern
# for i in range(num_lights_x):
#     for j in range(num_lights_y):
#         light_x = i * spacing_x - (num_lights_x / 2 * spacing_x)
#         light_y = j * spacing_y - (num_lights_y / 2 * spacing_y)
#         bpy.ops.object.light_add(type='AREA', location=( light_x-3,  light_y +6,  height_z))
#         point_light = bpy.context.object
#         point_light.data.energy = 80  # Adjust brightness of the point light
#         point_light.data.shadow_soft_size = 2  # Slightly soften the shadows

# [0.75, 8.3, -0.075],

bpy.ops.object.light_add(type='AREA', location=(-2, 7.5, 4))  # Position above the scene
area_light = bpy.context.object
area_light.data.energy = 1200  # Set intensity (increase for brighter indoor lighting)
area_light.data.size = 5  # Increase the size of the light to soften shadows

# # Add an area light (for broad, soft lighting)
# bpy.ops.object.light_add(type='AREA', location=(-5, 10, 0))  # Position above the scene
# area_light = bpy.context.object
# area_light.data.energy = 10000  # Set intensity (increase for brighter indoor lighting)
# area_light.data.size = 1  # Increase the size of the light to soften shadows

# # Add a point light (to simulate a smaller light source, like a light bulb)
# bpy.ops.object.light_add(type='POINT', location=(-5, 15, 2))  # Position as needed
# point_light = bpy.context.object
# point_light.data.energy = 10000  # Adjust brightness of the point light

############################ Soft Shadows with Area Lights ###############################

# # Add Area Light 1
# bpy.ops.object.light_add(type='AREA', location=(5, 5, 10))
# area_light_1 = bpy.context.active_object
# area_light_1.data.energy = 200  # Adjust brightness
# area_light_1.data.size = 5  # Increase size for softer shadows
# area_light_1.data.color = (1.0, 0.95, 0.9)  # Slightly warm light

# # Add Area Light 2
# bpy.ops.object.light_add(type='AREA', location=(-5, -5, 10))
# area_light_2 = bpy.context.active_object
# area_light_2.data.energy = 150  # Slightly dimmer
# area_light_2.data.size = 7  # Larger size for very soft shadows
# area_light_2.data.color = (0.9, 0.95, 1.0)  # Slightly cool light

# # Add Area Light 3 (backlight for fill)
# bpy.ops.object.light_add(type='AREA', location=(0, 0, 15))
# area_light_3 = bpy.context.active_object
# area_light_3.data.energy = 100  # Backfill light
# area_light_3.data.size = 10  # Very soft shadows
# area_light_3.data.color = (1.0, 1.0, 1.0)  # Neutral light
# area_light_3.rotation_euler = (radians(-90), 0, 0)  # Point downward

###########################################################################################





############################ hdri AS light sources ###############################

# # HDRI Lighting Setup
# if not bpy.context.scene.world:
#     bpy.context.scene.world = bpy.data.worlds.new("World")
# world = bpy.context.scene.world
# world.use_nodes = True

# env_nodes = world.node_tree.nodes
# env_links = world.node_tree.links

# # Clear existing world nodes
# for node in env_nodes:
#     env_nodes.remove(node)

# # Create and set up HDRI nodes
# env_texture_node = env_nodes.new(type="ShaderNodeTexEnvironment")
# background_node = env_nodes.new(type="ShaderNodeBackground")
# output_node = env_nodes.new(type="ShaderNodeOutputWorld")

# # Load HDRI image
# env_texture_node.image = bpy.data.images.load(hdri_path)
# env_texture_node.projection = 'EQUIRECTANGULAR'

# # Connect nodes
# env_links.new(env_texture_node.outputs["Color"], background_node.inputs["Color"])
# background_node.inputs["Strength"].default_value = sun_energy  # Adjust HDRI strength
# env_links.new(background_node.outputs["Background"], output_node.inputs["Surface"])

############################ hdri AS light sources ###############################




############################ hdri TO light sources ###############################

# # Load the HDRI using Blender's image handling
# hdr_image = bpy.data.images.load(hdri_path)

# # Convert HDRI data to numpy array for analysis
# hdr_data = np.array(hdr_image.pixels[:]).reshape(
#     hdr_image.size[1], hdr_image.size[0], 4
# )[:, :, :3]  # Extract RGB (ignore alpha)

# # Analyze the HDRI map
# brightness_map = np.sum(hdr_data, axis=2)
# threshold = np.percentile(brightness_map, 98)  # Top 2% brightest pixels
# binary_map = brightness_map > threshold

# # Cluster bright regions
# labeled_array, num_features = label(binary_map)

# # Calculate maximum intensity from HDRI for auto-scaling artificial light strength
# max_hdri_intensity = np.max(brightness_map)

# # Create lights based on cluster size and shape
# for i in range(1, num_features + 1):
#     region = np.argwhere(labeled_array == i)
#     y_coords, x_coords = region[:, 0], region[:, 1]
#     bbox_height = y_coords.max() - y_coords.min()
#     bbox_width = x_coords.max() - x_coords.min()
#     center_y = (y_coords.max() + y_coords.min()) // 2
#     center_x = (x_coords.max() + x_coords.min()) // 2

#     # Convert to spherical coordinates
#     u = center_x / hdr_data.shape[1]
#     v = center_y / hdr_data.shape[0]
#     theta = v * np.pi
#     phi = u * 2 * np.pi
#     radius = 15  # Distance of the light from the center
#     light_x = radius * np.sin(theta) * np.cos(phi)
#     light_y = radius * np.sin(theta) * np.sin(phi)
#     light_z = radius * np.cos(theta)

#     # Get color and normalize
#     light_color = hdr_data[center_y, center_x, :]
#     light_color /= np.max(light_color)  # Normalize RGB to [0, 1]

#     # Determine light type and intensity
#     light_intensity = (brightness_map[center_y, center_x] / max_hdri_intensity) * 100
#     light_intensity *= sun_energy

#     if bbox_width < 10 and bbox_height < 10:  # Small region
#         # Use point light
#         light_data = bpy.data.lights.new(name=f"PointLight_{i}", type='POINT')
#         light_data.energy = light_intensity
#         light_data.color = light_color
#         light_object = bpy.data.objects.new(name=f"PointLight_{i}", object_data=light_data)
#     elif bbox_width > bbox_height * 2 or bbox_height > bbox_width * 2:  # Long region
#         # Use tube light
#         light_data = bpy.data.lights.new(name=f"TubeLight_{i}", type='AREA')
#         light_data.shape = 'RECTANGLE'
#         light_data.size = bbox_width / 10
#         light_data.size_y = bbox_height / 10
#         light_data.energy = light_intensity
#         light_data.color = light_color
#         light_object = bpy.data.objects.new(name=f"TubeLight_{i}", object_data=light_data)
#     else:  # Diffuse, large region
#         # Use area light
#         light_data = bpy.data.lights.new(name=f"AreaLight_{i}", type='AREA')
#         light_data.size = max(bbox_width, bbox_height) / 10
#         light_data.energy = light_intensity
#         light_data.color = light_color
#         light_object = bpy.data.objects.new(name=f"AreaLight_{i}", object_data=light_data)

#     # Place the light
#     bpy.context.collection.objects.link(light_object)
#     light_object.location = Vector((light_x, light_y, light_z))


# # Remove lights for the next render
# # for obj in bpy.context.collection.objects:
# #     if obj.type == 'LIGHT':
# #         bpy.data.objects.remove(obj)
# # Ensure the scene has a world node tree
# if not bpy.context.scene.world:
#     bpy.context.scene.world = bpy.data.worlds.new("World")
# world = bpy.context.scene.world
# world.use_nodes = True
# # Set up HDRI environment lighting
# env_nodes = world.node_tree.nodes
# env_links = world.node_tree.links
# for node in env_nodes:
#     env_nodes.remove(node)
# background_node = env_nodes.new(type="ShaderNodeBackground")
# env_texture_node = env_nodes.new(type="ShaderNodeTexEnvironment")
# output_node = env_nodes.new(type="ShaderNodeOutputWorld")
# env_texture_node.image = bpy.data.images.load(hdri_path)
# env_links.new(env_texture_node.outputs["Color"], background_node.inputs["Color"])
# background_node.inputs["Strength"].default_value = 0.  # Adjust HDRI strength
# env_links.new(background_node.outputs["Background"], output_node.inputs["Surface"])



############################ hdri TO light sources ###############################



# Enable background image in camera view
camera.data.show_background_images = True
bg = camera.data.background_images.new()
bg.image = bg_image
bg.alpha = 1.0
bg.display_depth = 'BACK'
scene.view_settings.view_transform = 'Standard' 

# Use compositing nodes for background image placement
scene.use_nodes = True
tree = scene.node_tree
links = tree.links
for node in tree.nodes:
    tree.nodes.remove(node)

render_layers = tree.nodes.new(type='CompositorNodeRLayers')
bg_image_node = tree.nodes.new(type='CompositorNodeImage')
bg_image_node.image = bg_image

scale_node = tree.nodes.new(type='CompositorNodeScale')
scale_node.space = 'RENDER_SIZE'

alpha_over = tree.nodes.new(type='CompositorNodeAlphaOver')
composite = tree.nodes.new(type='CompositorNodeComposite')

links.new(bg_image_node.outputs['Image'], scale_node.inputs['Image'])
links.new(scale_node.outputs['Image'], alpha_over.inputs[1])
links.new(render_layers.outputs['Image'], alpha_over.inputs[2])
links.new(alpha_over.outputs['Image'], composite.inputs['Image'])

scene.render.film_transparent = True
scene.render.filepath = output_render_path
scene.render.image_settings.file_format = 'PNG'

bpy.ops.render.render(write_still=True)
