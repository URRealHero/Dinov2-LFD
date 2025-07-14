import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import trimesh
import numpy as np
import pyrender
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
import sys

def load_dinov2_model():
    """Loads the DinoV2 model and image processor from Hugging Face."""
    print("Loading DinoV2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    print(f"✅ DinoV2 model loaded on {device}.")
    return processor, model, device

def load_camera_positions(directory, num_rigs=10):
    """
    Loads camera positions from .obj files, structuring them by rig.
    This is required for the test.py script.
    """
    camera_rigs = []
    print(f"Loading {num_rigs} camera rigs from '{directory}'...")
    for i in range(num_rigs):
        cam_file = os.path.join(directory, f'12_{i}.obj')
        if os.path.exists(cam_file):
            # Each file contains vertices for one camera rig. We take the first 10.
            cam_mesh = trimesh.load(cam_file, force='mesh')
            camera_rigs.append(cam_mesh.vertices[:10])
    if not camera_rigs:
        print(f"❌ Error: No camera rigs found in '{directory}'.")
        sys.exit(1)
    print(f"✅ Loaded {len(camera_rigs)} camera rigs, each with {camera_rigs[0].shape[0]} views.")
    return camera_rigs

def create_view_matrix(camera_position, target, up_vector, distance=1):
    """
    Creates a view matrix (world-to-camera).
    This matrix positions the camera to look at the target from a specific point.
    """
    camera_position = camera_position * distance
    forward = target - camera_position
    if np.linalg.norm(forward) < 1e-6: return np.eye(4)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up_vector)
    if np.linalg.norm(right) < 1e-6:
        if abs(forward[1]) > 0.99:
            right = np.cross(forward, [0, 0, 1])
        else:
            right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    new_up = np.cross(right, forward)
    new_up /= np.linalg.norm(new_up)
    matrix = np.eye(4)
    matrix[0,:3], matrix[1,:3], matrix[2,:3] = right, new_up, -forward
    matrix[:3, 3] = -np.dot(matrix[:3,:3], camera_position)
    return matrix

def setup_scene(mesh):
    """
    Creates a pyrender scene with the model, lights, and a perspective camera.
    """
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.0, 0.0, 0.0, 1.0], # Black color
        metallicFactor=0.0,
        roughnessFactor=1.0
    )
    render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0])
    scene.add(render_mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, znear=0.1, zfar=100.0)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
    scene.add(camera, name='camera')
    scene.add(light, name='light')
    return scene

def extract_features(images, processor, model, device):
    """
    Extracts DinoV2 features from a list of PIL Images.
    """
    feature_list = []
    with torch.no_grad():
        for image in images:
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0].detach().cpu().numpy()
            feature_list.append(features)
    return np.vstack(feature_list)

def render_and_capture_views(mesh, camera_positions, renderer, distance=2):
    """
    Renders a mesh from multiple viewpoints and returns in-memory PIL images.
    This version does not save files, it just returns the image objects.
    """
    scene = setup_scene(mesh)
    camera_node = next(iter(scene.get_nodes(name='camera')))
    light_node  = next(iter(scene.get_nodes(name='light')))
    target, up = np.array([0, 0, 0]), np.array([0, 1, 0])
    captured_images = []
    for pos in camera_positions:
        view_matrix = create_view_matrix(pos, target, up, distance)
        cam_pose = np.linalg.inv(view_matrix)
        scene.set_pose(camera_node, cam_pose)
        scene.set_pose(light_node,  cam_pose)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        captured_images.append(Image.fromarray(color, 'RGBA'))
    return captured_images

def generate_features(mesh, camera_rigs, renderer, processor, model, device):
    """
    Generates a structured set of features (10, 10, 768) for a mesh.
    This corresponds to 10 LightField Descriptors and is used by test.py.
    """
    # --- Normalise Mesh ---
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    radius = np.linalg.norm(mesh.vertices, axis=1).max()
    if radius > 1e-6:
        mesh.apply_scale(1.0 / radius)

    all_descriptors = []
    for rig_cameras in tqdm(camera_rigs, desc="Encoding Descriptors"):
        # Render all 10 views for the current rig
        rendered_images = render_and_capture_views(mesh, rig_cameras, renderer)
        # Extract features for those 10 views to create one descriptor
        descriptor_features = extract_features(rendered_images, processor, model, device)
        all_descriptors.append(descriptor_features)
            
    return np.array(all_descriptors)

def save_features(features, output_file):
    """
    Saves the features to a file. Required by test.py.
    """
    print(f"Saving features to '{output_file}'...")
    np.save(output_file, features)
    print(f"✅ Features saved with shape {features.shape}.")

if __name__ == '__main__':
    # This main block is for simple rendering and saving of single feature sets,
    # not the full LFD generation which is handled by test.py
    # --- Configuration ---
    MODEL_PATHS = ['assets/model1.glb', 'assets/model2.glb']
    CAMERA_DATA_DIR = 'data'
    MAIN_OUTPUT_DIR = 'assets/render_images'
    # FEATURES_DIR = 'features'
    
    os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)
    # os.makedirs(FEATURES_DIR, exist_ok=True)

    # --- Setup ---
    cam_file = os.path.join(CAMERA_DATA_DIR, '12_0.obj')
    if not os.path.exists(cam_file):
        print(f"❌ Error: Camera file not found at '{cam_file}'")
        sys.exit(1)
        
    camera_positions = trimesh.load(cam_file, force='mesh').vertices[:10]
    renderer = pyrender.OffscreenRenderer(256, 256)
    # processor, model, device = load_dinov2_model()

    # --- Process Each Model ---
    for model_path in MODEL_PATHS:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_view_dir = os.path.join(MAIN_OUTPUT_DIR, f'{model_name}_views')
        os.makedirs(output_view_dir, exist_ok=True)
        
        print(f"\n--- Processing {model_path} ---")
        mesh = trimesh.load(model_path, force='mesh')
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        radius = np.linalg.norm(mesh.vertices, axis=1).max()
        mesh.apply_scale(1.0 / radius)

        # 1. Render views and save them to disk
        images = render_and_capture_views(mesh, camera_positions, renderer)
        print(f"Saving {len(images)} rendered views to '{output_view_dir}'...")
        for i, img in enumerate(images):
            img.save(os.path.join(output_view_dir, f"view_{i:02d}.png"))
        
        # # 2. Extract features from the in-memory images
        # features = extract_features(images, processor, model, device)
        
        # # 3. Save the features
        # feature_filename = os.path.join(FEATURES_DIR, f'{model_name}_features.npy')
        # np.save(feature_filename, features)
        # print(f"✅ Single feature set for {model_name} saved to {feature_filename} with shape {features.shape}")

    # --- Cleanup ---
    renderer.delete()
    print("\nDone.")
