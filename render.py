'''
Blender script to create synthetic images of object

'''

import bpy
import random
import os
import math
from mathutils import Vector

# configuration
output_path = r"/home/mayurf/synthetic_data/heavy_guy"
num_images = 1000
start_index = 1
target_object_name = "heavy_guy"
image_size = 640

# directory setup
image_folder = os.path.join(output_path, "images")
mask_folder = os.path.join(output_path, "masks")
os.makedirs(image_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)

# gpu setup
scene = bpy.context.scene
scene.render.engine = 'CYCLES'

prefs = bpy.context.preferences
cprefs = prefs.addons['cycles'].preferences
cprefs.refresh_devices()

cprefs.compute_device_type = 'CUDA'
cprefs.refresh_devices()
optix_available = any(d.type == 'OPTIX' for d in cprefs.devices)
if not optix_available:
    print("OptiX not available, falling back to CUDA")
    cprefs.compute_device_type = 'CUDA'
    cprefs.refresh_devices()

for device in cprefs.devices:
    device.use = device.type in ('OPTIX', 'CUDA')

scene.cycles.device = 'GPU'

# scene setup
scene.render.resolution_x = image_size
scene.render.resolution_y = image_size
scene.render.image_settings.file_format = 'PNG'
scene.render.film_transparent = True
scene.render.use_motion_blur = False
scene.view_layers[0].use_pass_object_index = True

# cycles qulaity
scene.cycles.samples = 128
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = 0.01
scene.cycles.adaptive_min_samples = 32

# memeor and performance stuff
scene.cycles.use_auto_tile = True
scene.cycles.tile_size = 2048
scene.cycles.debug_use_spatial_splits = True
scene.render.use_persistent_data = True

# light manupulation
scene.cycles.max_bounces = 4
scene.cycles.diffuse_bounces = 2
scene.cycles.glossy_bounces = 2
scene.cycles.transmission_bounces = 2
scene.cycles.shadow_terminator_offset = 0.1

# denosing disbaled
scene.cycles.use_denoising = False
scene.view_layers[0].cycles.use_denoising = False

# object setup
obj = bpy.data.objects.get(target_object_name)
if not obj:
    raise Exception(f"Object '{target_object_name}' not found.")
obj.pass_index = 1

# composisting setup
scene.use_nodes = True
tree = scene.node_tree
tree.nodes.clear()

for node in list(tree.nodes):
    if node.type == 'DENOISE':
        tree.nodes.remove(node)

rlayers     = tree.nodes.new("CompositorNodeRLayers")
composite   = tree.nodes.new("CompositorNodeComposite")
id_mask     = tree.nodes.new("CompositorNodeIDMask")
mask_output = tree.nodes.new("CompositorNodeOutputFile")

id_mask.index = 1
mask_output.format.file_format = 'PNG'
mask_output.base_path = mask_folder
mask_output.file_slots.clear()
mask_output.file_slots.new(name="mask")

tree.links.new(rlayers.outputs["IndexOB"], id_mask.inputs["ID value"])
tree.links.new(id_mask.outputs["Alpha"],   mask_output.inputs["mask"])
tree.links.new(rlayers.outputs["Image"],   composite.inputs["Image"])

# lights setup
if "Dynamic_Light" not in bpy.data.objects:
    light_data   = bpy.data.lights.new(name="Dynamic_Light", type='POINT')
    light_object = bpy.data.objects.new(name="Dynamic_Light", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
else:
    light_object = bpy.data.objects["Dynamic_Light"]

# camera setup
camera = bpy.data.objects.get("Camera")
if not camera:
    raise Exception("Camera object not found.")
camera.data.clip_start = 0.1
camera.data.clip_end   = 1000
sensor_height          = camera.data.sensor_height
sensor_width           = camera.data.sensor_width

# save default cam pose
base_cam_location = camera.location.copy()
base_cam_rotation = camera.rotation_euler.copy()

# object bounding sphere
bbox       = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
obj_center = sum(bbox, Vector()) / 8
obj_radius = max((v - obj_center).length for v in bbox)

print(f"Object center : {obj_center}")
print(f"Object radius : {obj_radius:.3f}")

# full object visibilty manupulation
def get_safe_distance(focal_length, sensor_w, sensor_h, radius, margin=1.15):
    fov_v   = 2 * math.atan(sensor_h / (2 * focal_length))
    fov_h   = 2 * math.atan(sensor_w / (2 * focal_length))
    fov_min = min(fov_v, fov_h)
    return (radius / math.sin(fov_min / 2)) * margin

def look_at(cam, target):
    direction = target - cam.location
    rot_quat  = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()


# azimuth, elevation, dist_multiplier, tilt_max

def shot_normal(i, n, min_dist):
    # Standard 360° coverage, mid elevation, moderate distance
    azimuth_deg   = ((i) / n) * 360.0 + random.uniform(-8, 8)
    elevation_deg = random.uniform(5, 50)
    dist          = random.uniform(min_dist * 1.0, min_dist * 1.3)
    tilt          = random.uniform(-5, 5)
    return azimuth_deg, elevation_deg, dist, tilt

def shot_close(min_dist):
    # Very close — almost uncomfortably tight on the object
    azimuth_deg   = random.uniform(0, 360)
    elevation_deg = random.uniform(-10, 60)
    dist          = random.uniform(min_dist * 0.85, min_dist * 1.05)
    tilt          = random.uniform(-8, 8)
    return azimuth_deg, elevation_deg, dist, tilt

def shot_far(min_dist):
    # Far back  object is small in frame but fully visible
    azimuth_deg   = random.uniform(0, 360)
    elevation_deg = random.uniform(0, 45)
    dist          = random.uniform(min_dist * 2.0, min_dist * 3.5)
    tilt          = random.uniform(-5, 5)
    return azimuth_deg, elevation_deg, dist, tilt

def shot_worm(min_dist):
    #Worm's eye camera very low, looking sharply upward.
    azimuth_deg   = random.uniform(0, 360)
    elevation_deg = random.uniform(-40, -15)   # below object center
    dist          = random.uniform(min_dist * 1.0, min_dist * 1.4)
    tilt          = random.uniform(-10, 10)
    return azimuth_deg, elevation_deg, dist, tilt

def shot_bird(min_dist):
    #bird's eye almost directly overhead.
    azimuth_deg   = random.uniform(0, 360)
    elevation_deg = random.uniform(72, 88)     # near top-down
    dist          = random.uniform(min_dist * 1.0, min_dist * 1.5)
    tilt          = random.uniform(-15, 15)
    return azimuth_deg, elevation_deg, dist, tilt

def shot_dutch(min_dist):
    #Dutch angle strong roll tilt for cinematic weirdness.
    azimuth_deg   = random.uniform(0, 360)
    elevation_deg = random.uniform(10, 55)
    dist          = random.uniform(min_dist * 1.0, min_dist * 1.6)
    tilt          = random.uniform(25, 45) * random.choice([-1, 1])  # heavy roll
    return azimuth_deg, elevation_deg, dist, tilt

def shot_extreme_close(min_dist):
    # extreme close-up  inside the safe zone, object fills frame
    azimuth_deg   = random.uniform(0, 360)
    elevation_deg = random.uniform(-15, 65)
    dist          = random.uniform(min_dist * 0.75, min_dist * 0.9)
    tilt          = random.uniform(-12, 12)
    return azimuth_deg, elevation_deg, dist, tilt

def shot_extreme_far(min_dist):
    #extreme far — very wide, object is tiny but visible.
    azimuth_deg   = random.uniform(0, 360)
    elevation_deg = random.uniform(0, 40)
    dist          = random.uniform(min_dist * 3.5, min_dist * 5.0)
    tilt          = random.uniform(-5, 5)
    return azimuth_deg, elevation_deg, dist, tilt

def shot_sideways(min_dist):
    #Camera rolled 90° — object appears sideways in frame.
    azimuth_deg   = random.uniform(0, 360)
    elevation_deg = random.uniform(0, 45)
    dist          = random.uniform(min_dist * 1.0, min_dist * 1.5)
    tilt          = random.uniform(75, 95) * random.choice([-1, 1])
    return azimuth_deg, elevation_deg, dist, tilt

def shot_diagonal_dive(min_dist):
    #High angle + strong tilt — like a diving surveillance camera
    azimuth_deg   = random.uniform(0, 360)
    elevation_deg = random.uniform(55, 75)
    dist          = random.uniform(min_dist * 1.0, min_dist * 1.8)
    tilt          = random.uniform(20, 40) * random.choice([-1, 1])
    return azimuth_deg, elevation_deg, dist, tilt

# shot distribution
SHOT_STYLES = [
    (shot_normal,         30),   # 30% — 360° coverage
    (shot_close,          12),   # 12% — tight close-ups
    (shot_far,            10),   # 10% — wide far shots
    (shot_worm,            8),   # 8%  — low worm's eye
    (shot_bird,            8),   # 8%  — overhead bird's eye
    (shot_dutch,           8),   # 8%  — dutch angle
    (shot_extreme_close,   8),   # 6%  — extreme close
    (shot_extreme_far,     6),   # 6%  — extreme far
    (shot_sideways,        6),   # 6%  — sideways roll
    (shot_diagonal_dive,   6),   # 6%  — diving high angle
]

# Build weighted list
style_pool = []
for fn, weight in SHOT_STYLES:
    style_pool.extend([fn] * weight)

#  main render loop
for i in range(start_index, start_index + num_images):
    idx = i - start_index
    print(f"\nn rendering {i}/{start_index + num_images - 1} ---")

    #fresh focal length each render
    focal_length     = random.uniform(24, 50)
    camera.data.lens = focal_length

    min_dist = get_safe_distance(focal_length, sensor_width, sensor_height, obj_radius)

    #pick shot style
    style_fn = random.choice(style_pool)

    # normal shot needs i and n for even azimuth spread
    if style_fn == shot_normal:
        azimuth_deg, elevation_deg, dist, tilt_deg = style_fn(idx, num_images, min_dist)
    else:
        azimuth_deg, elevation_deg, dist, tilt_deg = style_fn(min_dist)

    azimuth   = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)

    # Spherical  Cartesian (Z-up)
    cam_x = obj_center.x + dist * math.cos(elevation) * math.cos(azimuth)
    cam_y = obj_center.y + dist * math.cos(elevation) * math.sin(azimuth)
    cam_z = obj_center.z + dist * math.sin(elevation)

    camera.location = Vector((cam_x, cam_y, cam_z))

    # point at object center
    look_at(camera, obj_center)

    # apply roll tilt around Z (camera's own axis after look_at)
    camera.rotation_euler.rotate_axis('Z', math.radians(tilt_deg))

    # light
    light_object.location = obj_center + Vector((
        random.uniform(-obj_radius * 2.0, obj_radius * 2.0),
        random.uniform(-obj_radius * 2.0, obj_radius * 2.0),
        random.uniform(obj_radius * 0.5,  obj_radius * 3.5)
    ))
    light_object.data.energy = random.uniform(300, 1200)
    light_object.data.color  = (
        random.uniform(0.85, 1.0),
        random.uniform(0.85, 1.0),
        random.uniform(0.85, 1.0)
    )

    # output
    img_filename = f"image_{i:04d}.png"
    scene.render.filepath          = os.path.join(image_folder, img_filename)
    mask_output.file_slots[0].path = "image_"
    scene.frame_set(i)

    bpy.ops.render.render(write_still=True)
    print(f"✓ {img_filename} | style={style_fn.__name__:20s} | "
          f"az={azimuth_deg:6.1f}° el={elevation_deg:5.1f}° "
          f"dist={dist:.2f} tilt={tilt_deg:5.1f}°")

# reset camera
camera.location       = base_cam_location
camera.rotation_euler = base_cam_rotation
