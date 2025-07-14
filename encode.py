# encode.py
import trimesh
import numpy as np
import os
import pyrender
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# (load_dinov2_model, create_view_matrix, setup_scene, extract_features remain the same)
def load_dinov2_model():
    """Loads the DinoV2 model and image processor from Hugging Face."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    return processor, model, device

def load_camera_positions(directory, num_rigs=10):
    """Loads camera positions, keeping the structure of the rigs."""
    # This now returns a list of arrays, one for each rig
    camera_rigs = []
    for i in range(num_rigs):
        cam_file = os.path.join(directory, f'12_{i}.obj')
        if os.path.exists(cam_file):
            cam_mesh = trimesh.load(cam_file, force='mesh')
            # The paper uses 10 unique views per rig
            # We assume the first 10 vertices in the file are the unique views
            camera_rigs.append(cam_mesh.vertices[:10])
    print(f"✅ Loaded {len(camera_rigs)} camera rigs, each with {camera_rigs[0].shape[0]} views.")
    return camera_rigs

def generate_features(mesh, camera_rigs, renderer, processor, model, device):
    """
    Generates a structured set of features (10, 10, 768) for a mesh.
    This corresponds to 10 LightField Descriptors.
    """
    all_descriptors = []
    for rig_cameras in tqdm(camera_rigs, desc="Encoding Descriptors"):
        # This inner part is the same as before, but now runs for each rig
        scene = setup_scene(mesh)
        camera_node = scene.get_nodes(name='camera').pop()
        light_node = scene.get_nodes(name='light').pop()
        
        rendered_images = []
        target, up = [0, 0, 0], [0, 1, 0]
        for pos in rig_cameras:
            cam_pose = create_view_matrix(pos, target, up)
            scene.set_pose(camera_node, cam_pose)
            scene.set_pose(light_node, cam_pose)
            color, _ = renderer.render(scene)
            rendered_images.append(Image.fromarray(color))

        descriptor_features = extract_features(rendered_images, processor, model, device)
        all_descriptors.append(descriptor_features)
            
    return np.array(all_descriptors)

# (Helper functions like create_view_matrix, setup_scene, extract_features are unchanged)
def create_view_matrix(camera_position, target, up_vector):
    forward = target - camera_position
    if np.linalg.norm(forward) < 1e-6: return np.eye(4)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up_vector)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, [0, 0, 1]) if abs(forward[2]) < 0.99 else np.cross(forward, [1, 0, 0])
    right /= np.linalg.norm(right)
    new_up = np.cross(right, forward)
    new_up /= np.linalg.norm(new_up)
    matrix = np.eye(4)
    matrix[0,:3], matrix[1,:3], matrix[2,:3] = right, new_up, -forward
    matrix[:3, 3] = -np.dot(matrix[:3,:3], camera_position)
    return matrix
    
def setup_scene(mesh):
    render_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0])
    scene.add(render_mesh)
    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0, znear=0.01, zfar=10.0)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(camera, name='camera')
    scene.add(light, name='light')
    return scene

def extract_features(images, processor, model, device):
    feature_list = []
    with torch.no_grad():
        for image in images: # Removed tqdm here for cleaner output within the main loop
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0].detach().cpu().numpy()
            feature_list.append(features)
    return np.vstack(feature_list)